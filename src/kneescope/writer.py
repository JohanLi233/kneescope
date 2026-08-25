"""Reference writer for the kneescope snapshot format (see formats.py)."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .formats import (
    FORMAT_VERSION,
    GRAD_TEMPLATE,
    MANIFEST_NAME,
    MOM_TEMPLATE,
    validate_manifest,
)


def _default_created_by() -> str:
    try:
        return f"kneescope {_pkg_version('kneescope')}"
    except PackageNotFoundError:
        return "kneescope (uninstalled)"


def _check_path_key(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError("parameter paths must be non-empty strings")
    if path.startswith("__"):
        raise ValueError(f"parameter path {path!r} uses the reserved '__' prefix")
    if "/" in path or "\\" in path or "\x00" in path:
        raise ValueError(f"parameter path {path!r} is not a valid npz key")


class SnapshotWriter:
    """Writes momentum and micro-batch gradient snapshots in format v1.

    Parameters
    ----------
    root:
        Output directory (created if needed). If a manifest already exists it
        is resumed: previously recorded steps and micro-batch counts are kept.
    notes:
        Free-form dict stored in the manifest (e.g. run id, config hash).
    created_by:
        Provenance string; defaults to ``"kneescope <version>"``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        notes: Mapping[str, object] | None = None,
        created_by: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._created_by = created_by or _default_created_by()
        self._notes: dict[str, object] = dict(notes or {})
        self._steps: set[int] = set()
        self._mb_counts: dict[int, int] = {}
        mpath = self.root / MANIFEST_NAME
        if mpath.is_file():
            manifest = json.loads(mpath.read_text())
            validate_manifest(manifest)
            self._steps.update(int(s) for s in manifest["steps"])
            for step in self._steps:
                n = len(list(self.root.glob(f"grad_step{step}_mb*.npz")))
                if n:
                    self._mb_counts[step] = n

    # -- structured subsampling of stacked tensors -------------------------

    @staticmethod
    def subsample_indices(n: int, k: int | None) -> list[int]:
        """Evenly-spaced indices into ``range(n)``, keeping at most ``k``.

        Structured (not random) subsampling: the same rule applied to momentum
        and gradients of a stacked tensor selects the same slices, which the
        estimator relies on.
        """
        if k is None or k >= n:
            return list(range(n))
        if k < 1:
            raise ValueError("subsample count must be >= 1")
        idx = np.linspace(0, n - 1, k).astype(int)
        return sorted(set(idx.tolist()))

    @staticmethod
    def split_stacked(
        path: str,
        array: ArrayLike,
        max_slices: int | None = None,
    ) -> dict[str, NDArray]:
        """Split a stacked tensor ``(..., r, c)`` into named 2D matrices.

        Leading dimensions are flattened into a single stack axis; slice ``i``
        gets the synthesized path ``f"{path}[e{i}]"`` (e.g.
        ``blocks.3.mlp.experts.gu[e37]``). With ``max_slices`` set, an evenly
        spaced structured subset of slices is kept (see
        :meth:`subsample_indices`). 2D input is returned unchanged under
        ``path``.
        """
        arr = np.asarray(array)
        if arr.ndim < 2:
            raise ValueError(f"{path}: expected at least 2 dims, got {arr.shape}")
        if arr.ndim == 2:
            return {path: arr}
        n_stack = int(np.prod(arr.shape[:-2]))
        flat = arr.reshape(n_stack, *arr.shape[-2:])
        return {
            f"{path}[e{i}]": flat[i]
            for i in SnapshotWriter.subsample_indices(n_stack, max_slices)
        }

    # -- writing ------------------------------------------------------------

    def write_momentum(self, step: int, matrices: Mapping[str, ArrayLike]) -> Path:
        """Write ``mom_step{step}.npz``: the momentum fed into orthogonalization."""
        out = self.root / MOM_TEMPLATE.format(step=step)
        self._write_npz(out, matrices)
        self._steps.add(int(step))
        self._sync_manifest()
        return out

    def write_grad_microbatch(
        self, step: int, index: int, matrices: Mapping[str, ArrayLike]
    ) -> Path:
        """Write ``grad_step{step}_mb{index}.npz``: one micro-batch gradient.

        ``matrices`` must hold the gradient at its true per-microbatch scale
        (not divided by the accumulation count), with the same paths as the
        momentum snapshot.
        """
        if index < 0:
            raise ValueError("micro-batch index must be >= 0")
        out = self.root / GRAD_TEMPLATE.format(step=step, index=index)
        self._write_npz(out, matrices)
        self._mb_counts[int(step)] = max(self._mb_counts.get(int(step), 0), index + 1)
        self._sync_manifest()
        return out

    def _write_npz(self, out: Path, matrices: Mapping[str, ArrayLike]) -> None:
        payload: dict[str, np.ndarray] = {}
        for path, arr in matrices.items():
            _check_path_key(path)
            a = np.asarray(arr)
            if a.ndim != 2:
                raise ValueError(
                    f"{path}: snapshot entries must be 2D, got {a.shape}; "
                    "split stacked tensors with SnapshotWriter.split_stacked first"
                )
            payload[path] = a.astype(np.float32)
        np.savez(out, **payload)

    def _sync_manifest(self) -> None:
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_by": self._created_by,
            "steps": sorted(self._steps),
            "n_microbatches": max(self._mb_counts.values(), default=0),
            "notes": self._notes,
        }
        (self.root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
