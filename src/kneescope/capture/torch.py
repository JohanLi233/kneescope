"""PyTorch capture adapter for kneescope.

torch is needed only at capture time; importing this module (and the base
package) stays numpy-only. Tensors are converted via duck-typing
(``.detach().float().cpu().numpy()``), so any object with the tensor API works.

Three lines added to an existing training loop::

    probe = TorchKneeProbe(model, steps={300, 600}, out_dir="knees", optimizer=opt)
    ...
    probe.begin_microbatches(step)                    # opens the capture window on probe steps
    for micro in micro_batches:
        loss = model(micro) / accum_steps
        with probe.capture_microbatch(loss):          # stashes grads after each backward
            loss.backward()
    probe.maybe_snapshot_momentum(step)               # right before opt.step()
    opt.step()

Notes
-----
* ``maybe_snapshot_momentum(step)`` captures the momentum buffer *before* the
  optimizer's orthogonalization/update at the listed steps (post-EMA/nesterov,
  pre-normalization — exactly what the estimator expects).
* Micro-batch gradients are taken from ``param.grad`` as-is. Make sure your
  loop zeroes gradients between micro-batches (or uses hooks) so each record
  is one fresh micro-batch, not an accumulation.
* The capture window opened by ``begin_microbatches(step)`` stays open across
  optimizer updates until ``n_microbatches`` have been recorded, then flushes
  to disk and closes automatically.
* Memory discipline: nothing is captured off the listed probe steps; tensors
  are converted to float32 CPU numpy immediately; large stacked expert tensors
  can be structured-subsampled via ``subsample``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

from ..writer import SnapshotWriter


def _to_numpy(t: Any) -> np.ndarray:
    """Detach a tensor-like object to a float32 CPU numpy array (duck-typed)."""
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "float"):
        t = t.float()
    if hasattr(t, "cpu"):
        t = t.cpu()
    return np.asarray(t.numpy() if hasattr(t, "numpy") else t, dtype=np.float32)


def _iter_named(model_or_params: Any) -> Iterator[tuple[str, Any]]:
    """Yield (name, param) from a model, a mapping, or an iterable."""
    if hasattr(model_or_params, "named_parameters"):
        yield from model_or_params.named_parameters()
    elif isinstance(model_or_params, Mapping):
        yield from model_or_params.items()
    else:
        for i, item in enumerate(model_or_params):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                yield item
            else:
                yield f"param_{i}", item


class TorchKneeProbe:
    """Minimal-integration probe that writes kneescope snapshots from PyTorch.

    Parameters
    ----------
    model_or_params:
        A ``torch.nn.Module`` (uses ``named_parameters()``), a mapping of
        name -> parameter, or an iterable of parameters / (name, parameter)
        pairs. Only parameters with ndim >= 2 are probed (the Muon group).
    steps:
        Optimizer steps to probe (the step convention is yours — pass the same
        counter to ``begin_microbatches`` / ``maybe_snapshot_momentum``).
    out_dir:
        Snapshot output directory (format v1, see :mod:`kneescope.formats`).
    momentum_getter:
        ``callable(param) -> 2D tensor | None`` returning the momentum matrix
        fed into the orthogonalization. Adapts any Muon variant's state layout.
        Default (via ``optimizer=`` or :meth:`attach_optimizer`) reads
        ``optimizer.state[param]["momentum_buffer"]``.
    subsample:
        If set, keep at most this many evenly-spaced slices from the flattened
        leading (stack) dims of >2D tensors such as stacked experts; slices
        become ``path[e{i}]`` entries. Momentum and gradients use the same
        deterministic subset.
    n_microbatches:
        Micro-batch gradients captured per probed step (>= 2).
    optimizer:
        Convenience: a torch optimizer to build the default momentum getter.
    state_key:
        Optimizer-state entry the default getter reads
        (default ``"momentum_buffer"``).
    """

    def __init__(
        self,
        model_or_params: Any,
        steps: Iterable[int],
        out_dir: str | Path,
        momentum_getter: Callable[[Any], Any] | None = None,
        subsample: int | None = None,
        n_microbatches: int = 4,
        optimizer: Any | None = None,
        state_key: str = "momentum_buffer",
    ) -> None:
        if n_microbatches < 2:
            raise ValueError("n_microbatches must be >= 2")
        self.steps = frozenset(int(s) for s in steps)
        self.writer = SnapshotWriter(out_dir)
        self.subsample = subsample
        self.n_microbatches = n_microbatches
        self.momentum_getter = momentum_getter
        if optimizer is not None:
            self.attach_optimizer(optimizer, state_key=state_key)
        # Materialize the parameter list once; params are stable across steps.
        self._named = [
            (name, p)
            for name, p in _iter_named(model_or_params)
            if int(getattr(p, "ndim", len(getattr(p, "shape", (1,))))) >= 2
        ]
        self._mb_step: int | None = None
        self._mb_stash: list[dict[str, np.ndarray]] = []

    # -- momentum -----------------------------------------------------------

    def attach_optimizer(self, optimizer: Any, state_key: str = "momentum_buffer") -> None:
        """Set the default momentum getter from a torch optimizer's state."""

        def getter(param: Any) -> Any:
            state = getattr(optimizer, "state", {})
            try:
                entry = state.get(param)
            except AttributeError:
                entry = None
            if isinstance(entry, Mapping):
                return entry.get(state_key)
            return None

        self.momentum_getter = getter

    def maybe_snapshot_momentum(self, step: int) -> bool:
        """Write the momentum snapshot if ``step`` is a probe step.

        Call right before the optimizer's orthogonalization/update. No-op
        (returns False) outside probe steps.
        """
        if step not in self.steps:
            return False
        if self.momentum_getter is None:
            raise RuntimeError(
                "no momentum_getter: pass momentum_getter=... to the constructor, "
                "or call attach_optimizer(optimizer) to read "
                "optimizer.state[param]['momentum_buffer']"
            )
        mats: dict[str, np.ndarray] = {}
        for name, p in self._named:
            buf = self.momentum_getter(p)
            if buf is None:
                continue
            self._collect(mats, name, buf)
        self.writer.write_momentum(step, mats)
        return True

    # -- micro-batch gradients ------------------------------------------------

    def begin_microbatches(self, step: int) -> bool:
        """Open the gradient capture window for a probe step.

        Returns False (and closes any open window) off probe steps. An open
        window stays open across optimizer updates until ``n_microbatches``
        have been recorded, then flushes to disk and closes itself.
        """
        if step not in self.steps:
            self._mb_step = None
            self._mb_stash = []
            return False
        self._mb_step = int(step)
        self._mb_stash = []
        return True

    def record_microbatch(self) -> int:
        """Stash the current ``param.grad`` clones as one micro-batch.

        Call after each backward pass during the probed window. Returns the
        number of micro-batches stashed so far (0 when no window is open or
        the window is already full).
        """
        if self._mb_step is None or len(self._mb_stash) >= self.n_microbatches:
            return 0
        mats: dict[str, np.ndarray] = {}
        for name, p in self._named:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            self._collect(mats, name, g)
        self._mb_stash.append(mats)
        if len(self._mb_stash) == self.n_microbatches:
            for j, stash in enumerate(self._mb_stash):
                self.writer.write_grad_microbatch(self._mb_step, j, stash)
            self._mb_step = None
            self._mb_stash = []
        return len(self._mb_stash)

    @contextlib.contextmanager
    def capture_microbatch(self, loss: Any = None) -> Iterator[Any]:
        """Context-manager form of :meth:`record_microbatch`.

        Usage: ``with probe.capture_microbatch(loss): loss.backward()`` —
        gradients are stashed on exit. No-op outside the probed window.
        """
        yield loss
        self.record_microbatch()

    # -- internals ------------------------------------------------------------

    def _collect(self, out: dict[str, np.ndarray], name: str, tensor: Any) -> None:
        arr = _to_numpy(tensor)
        if arr.ndim == 2:
            out[name] = arr
        else:
            out.update(
                SnapshotWriter.split_stacked(name, arr, max_slices=self.subsample)
            )
