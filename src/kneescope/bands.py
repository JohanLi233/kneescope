"""Log-sigma binning, persistence-band detection, and the top-level analyze().

Per-direction persistence values (rho, floor) are pooled per *shape group*
(all matrices of the same ``(rows, cols)``) and binned by log sigma. Per bin
the MEDIAN rho and MEDIAN floor are taken — the median is essential because
the floor blows up at tiny sigma — and ``excess = median_rho - median_floor``.

Operational band definitions (thresholds configurable):

- ``sigma*``       = lower edge of the first bin (from low sigma) with
  ``excess >= low`` (default 0.3),
- ``sigma*_upper`` = lower edge of the first bin with ``excess >= high``
  (default 0.9).

Honesty rule: a crossing is only reported if at least one populated bin below
it sits under the threshold. If signal is already persistent at the lowest
visible sigma, or no bin reaches the threshold, the group is reported as
"no collapse in view" (``sigma* is None``) — never an invented number.

For a single-knee schedule the recommended knee50 is ``sigma*_upper``, NOT
``sigma*`` (see the companion paper "The Persistence Boundary", sec. 8).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .estimator import MatrixEstimate, analyze_matrix, anisotropy_ratios
from .formats import Snapshot, load_snapshot

DEFAULT_BIN_EDGES: tuple[float, ...] = (
    1e-5, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0,
)

DEFAULT_MIN_PER_BIN = 10
DEFAULT_LOW_THRESHOLD = 0.3
DEFAULT_HIGH_THRESHOLD = 0.9


class BoundaryStatus(enum.Enum):
    """Why a boundary value is or is not reported."""

    FOUND = "found"
    BELOW_VIEW = (
        "no collapse in view: signal already persistent at the lowest visible sigma"
    )
    ABOVE_VIEW = "no collapse in view: no visible bin reaches the threshold"
    NO_DATA = "no collapse in view: no populated bins"


@dataclass
class BinStats:
    """One log-sigma bin of pooled per-direction persistence values."""

    lo: float
    hi: float
    n: int
    median_rho: float
    median_floor: float

    @property
    def excess(self) -> float:
        return self.median_rho - self.median_floor


@dataclass
class BandReport:
    """Persistence band for one shape group at one optimizer step."""

    shape: tuple[int, int]
    step: int | None
    n_matrices: int
    n_directions: int
    bins: list[BinStats]
    sigma_star: float | None
    sigma_star_upper: float | None
    star_status: BoundaryStatus
    upper_status: BoundaryStatus
    low_threshold: float
    high_threshold: float
    anisotropy_row: float
    anisotropy_col: float
    zipf_median: float
    spectrum: dict[str, float] = field(default_factory=dict)
    mp: object | None = None  # mp_diag.MpGroupSummary, only when include_mp=True

    @property
    def recommended_knee50(self) -> float | None:
        """Where a single-knee schedule should place knee50: the UPPER band edge.

        None when no upper boundary was found in the visible window.
        """
        return self.sigma_star_upper


@dataclass
class AnalysisResult:
    """All shape-group reports for a snapshot, keyed by ``(step, shape)``."""

    groups: dict[tuple[int | None, tuple[int, int]], BandReport]
    low: float = DEFAULT_LOW_THRESHOLD
    high: float = DEFAULT_HIGH_THRESHOLD
    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES
    warnings: list[str] = field(default_factory=list)

    def text(self) -> str:
        from .report import format_report

        return format_report(self)

    def plot(self, path: str | Path | None = None, *, show: bool = False):
        """Plot median rho(sigma) per group with the transition band shaded.

        Requires matplotlib (``pip install kneescope[matplotlib]``).
        """
        from .report import plot_result

        return plot_result(self, path=path, show=show)

    def __str__(self) -> str:
        return self.text()


def bin_directions(
    sigma: np.ndarray,
    rho: np.ndarray,
    floor: np.ndarray,
    edges: Sequence[float] = DEFAULT_BIN_EDGES,
    min_count: int = DEFAULT_MIN_PER_BIN,
) -> list[BinStats]:
    """Pool per-direction values into log-sigma bins (median rho / median floor).

    Bins with fewer than ``min_count`` directions are skipped. The top bin is
    closed on the right so sigma == 1.0 is not dropped.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    edges = np.asarray(tuple(edges), dtype=np.float64)
    bins: list[BinStats] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        msk = (sigma >= lo) & (sigma <= hi if last else sigma < hi)
        n = int(msk.sum())
        if n < min_count:
            continue
        bins.append(
            BinStats(
                lo=float(lo),
                hi=float(hi),
                n=n,
                median_rho=float(np.median(rho[msk])),
                median_floor=float(np.median(floor[msk])),
            )
        )
    return bins


def _first_crossing(
    bins: Sequence[BinStats], threshold: float
) -> tuple[float | None, BoundaryStatus]:
    """Lower edge of the first bin (from low sigma) with excess >= threshold.

    Only counts as a crossing if at least one earlier populated bin sits below
    the threshold — otherwise the transition is outside the visible window and
    no number is invented.
    """
    if not bins:
        return None, BoundaryStatus.NO_DATA
    seen_below = False
    for b in bins:
        if b.excess >= threshold:
            if seen_below:
                return b.lo, BoundaryStatus.FOUND
        else:
            seen_below = True
    return None, (
        BoundaryStatus.ABOVE_VIEW if seen_below else BoundaryStatus.BELOW_VIEW
    )


def detect_band(
    bins: Sequence[BinStats],
    low: float = DEFAULT_LOW_THRESHOLD,
    high: float = DEFAULT_HIGH_THRESHOLD,
) -> tuple[float | None, float | None, BoundaryStatus, BoundaryStatus]:
    """Detect (sigma*, sigma*_upper, statuses) from populated bins."""
    star, star_status = _first_crossing(bins, low)
    upper, upper_status = _first_crossing(bins, high)
    return star, upper, star_status, upper_status


def _spectrum_quantiles(oks: Sequence[MatrixEstimate]) -> dict[str, float]:
    rows = []
    for e in oks:
        S = e.sigma
        k = S.size
        rows.append(
            [
                S[0],
                S[max(0, int(0.05 * k))],
                S[int(0.25 * k)],
                S[int(0.5 * k)],
                S[int(0.75 * k)],
                S[-1],
            ]
        )
    q = np.median(np.asarray(rows), axis=0)
    return {
        "sigma1": float(q[0]),
        "sigma05": float(q[1]),
        "sigma25": float(q[2]),
        "sigma50": float(q[3]),
        "sigma75": float(q[4]),
        "sigma_min": float(q[5]),
    }


def build_report(
    estimates: Sequence[MatrixEstimate],
    *,
    shape: tuple[int, int],
    step: int | None = None,
    edges: Sequence[float] = DEFAULT_BIN_EDGES,
    min_count: int = DEFAULT_MIN_PER_BIN,
    low: float = DEFAULT_LOW_THRESHOLD,
    high: float = DEFAULT_HIGH_THRESHOLD,
    include_mp: bool = False,
) -> BandReport:
    """Aggregate per-matrix estimates of one shape group into a BandReport."""
    oks = [e for e in estimates if e.ok]
    bins: list[BinStats] = []
    a_row = a_col = zipf = float("nan")
    spectrum: dict[str, float] = {}
    n_dirs = 0
    if oks:
        sigma = np.concatenate([e.sigma for e in oks])
        rho = np.concatenate([e.rho for e in oks])
        floor = np.concatenate([e.rho_floor for e in oks])
        n_dirs = int(sigma.size)
        bins = bin_directions(sigma, rho, floor, edges, min_count)
        m, n = shape
        a_row, a_col = anisotropy_ratios(
            np.concatenate([e.row_noise for e in oks]),
            np.concatenate([e.col_noise for e in oks]),
            m,
            n,
        )
        zipf_vals = [e.zipf_slope for e in oks if np.isfinite(e.zipf_slope)]
        zipf = float(np.median(zipf_vals)) if zipf_vals else float("nan")
        spectrum = _spectrum_quantiles(oks)
    star, upper, star_status, upper_status = detect_band(bins, low, high)
    mp = None
    if include_mp:
        from . import mp_diag

        mp = mp_diag.summarize_group(oks, shape)
    return BandReport(
        shape=shape,
        step=step,
        n_matrices=len(estimates),
        n_directions=n_dirs,
        bins=bins,
        sigma_star=star,
        sigma_star_upper=upper,
        star_status=star_status,
        upper_status=upper_status,
        low_threshold=low,
        high_threshold=high,
        anisotropy_row=a_row,
        anisotropy_col=a_col,
        zipf_median=zipf,
        spectrum=spectrum,
        mp=mp,
    )


def analyze(
    source: str | Path | Snapshot,
    *,
    steps: Sequence[int] | None = None,
    low: float = DEFAULT_LOW_THRESHOLD,
    high: float = DEFAULT_HIGH_THRESHOLD,
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
    include_mp: bool = False,
) -> AnalysisResult:
    """Run the persistence analysis over a snapshot.

    Parameters
    ----------
    source:
        Snapshot directory path, or an in-memory :class:`Snapshot`.
    steps:
        Restrict to these optimizer steps (default: all steps present).
    low, high:
        Excess thresholds for sigma* and sigma*_upper.
    bin_edges:
        Log-sigma bin edges; the default spans [1e-5, 1.0].
    min_per_bin:
        Minimum pooled directions for a bin to be reported.
    include_mp:
        Also compute the Marchenko–Pastur diagnostic per group (assumption-
        laden; see :mod:`kneescope.mp_diag`).

    Returns
    -------
    AnalysisResult
        Groups keyed by ``(step, (rows, cols))``; call ``.text()`` for the
        report or ``.plot()`` for the rho(sigma) curves.
    """
    if isinstance(source, (str, Path)):
        snap = load_snapshot(source, steps=list(steps) if steps is not None else None)
    elif isinstance(source, Snapshot):
        snap = source
    else:
        raise TypeError(
            f"source must be a path or a Snapshot, got {type(source).__name__}"
        )

    warnings: list[str] = []
    groups: dict[tuple[int | None, tuple[int, int]], BandReport] = {}
    want = sorted(snap.steps) if steps is None else list(steps)
    for step in want:
        moms = snap.momentum.get(step)
        if not moms:
            warnings.append(f"step {step}: no momentum snapshot; skipped")
            continue
        glist = snap.grads.get(step, [])
        if len(glist) < 2:
            warnings.append(
                f"step {step}: fewer than 2 micro-batch gradient snapshots; skipped"
            )
            continue
        by_shape: dict[tuple[int, int], list[tuple[str, np.ndarray, list]]] = {}
        for path, mat in sorted(moms.items()):
            mat = np.asarray(mat)
            if mat.ndim != 2:
                warnings.append(f"step {step}: {path}: not 2D ({mat.shape}); skipped")
                continue
            gs = [g[path] for g in glist if path in g]
            if len(gs) < 2:
                warnings.append(
                    f"step {step}: {path}: present in {len(gs)} micro-batch "
                    "snapshots (< 2); skipped"
                )
                continue
            if gs[0].shape != mat.shape:
                warnings.append(
                    f"step {step}: {path}: grad shape {gs[0].shape} != momentum "
                    f"shape {mat.shape}; skipped"
                )
                continue
            by_shape.setdefault(mat.shape, []).append((path, mat, gs))
        for shape, items in sorted(by_shape.items()):
            ests = [analyze_matrix(mat, gs) for _, mat, gs in items]
            groups[(step, shape)] = build_report(
                ests,
                shape=shape,
                step=step,
                edges=bin_edges,
                min_count=min_per_bin,
                low=low,
                high=high,
                include_mp=include_mp,
            )
    return AnalysisResult(
        groups=groups, low=low, high=high, bin_edges=tuple(bin_edges), warnings=warnings
    )
