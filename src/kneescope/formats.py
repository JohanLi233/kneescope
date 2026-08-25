"""Snapshot format v1: on-disk layout, validating reader.

Directory layout::

    <snapdir>/
      manifest.json
      mom_step{k}.npz          one entry per parameter path: the 2D momentum
                               matrix FED INTO the orthogonalization
                               (post-EMA/nesterov, pre-normalization), float32
      grad_step{k}_mb{j}.npz   same paths; micro-batch gradient j at its true
                               per-microbatch scale (NOT divided by the
                               accumulation count)

``manifest.json``::

    {
      "format_version": 1,
      "created_by": "kneescope 0.1.0",
      "steps": [int, ...],        // optimizer steps with a momentum snapshot
      "n_microbatches": int,      // micro-batches captured per probed step
      "notes": {...}              // free-form
    }

npz keys are parameter paths, e.g. ``blocks.3.attn.wq.weight``. Stacked expert
tensors must be split by the capturing adapter into individual 2D matrices with
synthesized paths such as ``blocks.3.mlp.experts.gu[e37]`` (see
``SnapshotWriter.split_stacked``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MOM_TEMPLATE = "mom_step{step}.npz"
GRAD_TEMPLATE = "grad_step{step}_mb{index}.npz"
_GRAD_GLOB = "grad_step{step}_mb*.npz"
_MB_RE = re.compile(r"grad_step-?\d+_mb(\d+)\.npz$")

_MANIFEST_REQUIRED = ("format_version", "created_by", "steps", "n_microbatches", "notes")


class SnapshotFormatError(ValueError):
    """Raised when a snapshot directory violates the format spec."""


def validate_manifest(manifest: Any) -> None:
    """Raise :class:`SnapshotFormatError` unless ``manifest`` is a valid v1 manifest."""
    if not isinstance(manifest, dict):
        raise SnapshotFormatError("manifest.json must contain a JSON object")
    for key in _MANIFEST_REQUIRED:
        if key not in manifest:
            raise SnapshotFormatError(f"manifest.json: missing required key {key!r}")
    if manifest["format_version"] != FORMAT_VERSION:
        raise SnapshotFormatError(
            f"unsupported format_version {manifest['format_version']!r}; "
            f"this kneescope reads version {FORMAT_VERSION}"
        )
    steps = manifest["steps"]
    if not isinstance(steps, list) or not all(isinstance(s, int) for s in steps):
        raise SnapshotFormatError("manifest.json: 'steps' must be a list of ints")
    n_mb = manifest["n_microbatches"]
    if not isinstance(n_mb, int) or n_mb < 0:
        raise SnapshotFormatError("manifest.json: 'n_microbatches' must be an int >= 0")
    if not isinstance(manifest["notes"], dict):
        raise SnapshotFormatError("manifest.json: 'notes' must be an object")


@dataclass
class Snapshot:
    """In-memory snapshot: momentum and per-microbatch gradients per step.

    ``grads[step]`` is a list of ``{path: 2D array}`` dicts ordered by
    micro-batch index. Can be built directly (bypassing the disk format) for
    tests or custom pipelines.
    """

    momentum: dict[int, dict[str, np.ndarray]]
    grads: dict[int, list[dict[str, np.ndarray]]]
    manifest: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    @property
    def steps(self) -> list[int]:
        return sorted(self.momentum)

    def n_microbatches(self, step: int) -> int:
        return len(self.grads.get(step, []))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _mb_index(path: Path) -> int:
    m = _MB_RE.search(path.name)
    return int(m.group(1)) if m else -1


def load_snapshot(
    root: str | Path,
    *,
    steps: list[int] | None = None,
    strict: bool = True,
) -> Snapshot:
    """Load a snapshot directory, validating the manifest.

    Parameters
    ----------
    root:
        Snapshot directory containing ``manifest.json``.
    steps:
        Restrict to these optimizer steps (default: all steps in the manifest).
    strict:
        If True (default), raise :class:`SnapshotFormatError` when a step
        listed in the manifest has no momentum file, or fewer gradient
        micro-batch files than ``n_microbatches``. If False, skip affected
        steps silently.
    """
    root = Path(root)
    mpath = root / MANIFEST_NAME
    if not mpath.is_file():
        raise SnapshotFormatError(f"{root}: missing {MANIFEST_NAME}")
    try:
        manifest = json.loads(mpath.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotFormatError(f"{mpath}: invalid JSON: {exc}") from exc
    validate_manifest(manifest)

    want = list(manifest["steps"]) if steps is None else list(steps)
    momentum: dict[int, dict[str, np.ndarray]] = {}
    grads: dict[int, list[dict[str, np.ndarray]]] = {}
    for step in want:
        mfile = root / MOM_TEMPLATE.format(step=step)
        if not mfile.is_file():
            if strict:
                raise SnapshotFormatError(
                    f"{root}: manifest lists step {step} but {mfile.name} is missing"
                )
            continue
        gfiles = sorted(root.glob(_GRAD_GLOB.format(step=step)), key=_mb_index)
        if strict and len(gfiles) < manifest["n_microbatches"]:
            raise SnapshotFormatError(
                f"{root}: step {step} has {len(gfiles)} gradient micro-batch files, "
                f"manifest declares n_microbatches={manifest['n_microbatches']}"
            )
        momentum[step] = _load_npz(mfile)
        grads[step] = [_load_npz(g) for g in gfiles]
    return Snapshot(momentum=momentum, grads=grads, manifest=manifest, root=root)
