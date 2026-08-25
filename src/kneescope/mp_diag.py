"""OPTIONAL Marchenko-Pastur / lambda+ diagnostic — assumption-laden.

.. warning::
   Everything in this module assumes the noise part of the momentum spectrum
   follows the Marchenko-Pastur (MP) law with iid entries plus a finite number
   of spikes (the BBP/BGN spiked model). The companion paper ("The Persistence
   Boundary") shows this calibration FAILS on real training momentum, whose
   spectra are heavy-tailed; that is why the tool's core estimator
   (:mod:`kneescope.estimator`) is assumption-free cross-microbatch
   persistence. Use this module only as a diagnostic, and distrust it whenever
   the Zipf slope indicates heavy tails or the KS check rejects MP.

Conventions (F-normalized spectrum, ``sum(sigma^2) = 1``): the iid noise part
``E`` of an ``m x n`` matrix has singular values following the MP law with
ratio ``c = max(m, n) / min(m, n) >= 1``; the bulk upper edge is
``lam_plus = sqrt(w) * (1/sqrt(m) + 1/sqrt(n))`` with ``w`` the noise
Frobenius-energy share. The BGN (Benaych-Georges-Nadakuditi 2012) formulas
give the observed singular value and singular-vector overlaps of a spike of
strength ``theta`` (in units where the noise entries have variance ``1/n``,
``n`` the short side).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .estimator import MatrixEstimate


def lam_plus(m: int, n: int, w: float) -> float:
    """MP bulk upper edge in F-normalized units: sqrt(w) (1/sqrt(m) + 1/sqrt(n))."""
    return float(np.sqrt(w) * (1.0 / np.sqrt(max(m, n)) + 1.0 / np.sqrt(min(m, n))))


def peel_w_c(sigma: NDArray[np.floating], c: float, *, iters: int = 10) -> tuple[float, int, bool]:
    """Estimate the noise energy share ``w`` by spike peeling.

    Iterates ``w <- 1 - sum(sigma^2 over sigma > lam_plus(w))`` to a fixed
    point. ``c = max/min >= 1``; ``sigma`` must be descending, F-normalized.

    Returns ``(w, n_spikes, ok)``. ``ok`` is False when nearly the whole
    spectrum sits above the edge (not an MP+spikes shape — peeling diverged)
    or the estimated noise share collapsed to ~0.
    """
    sig = np.asarray(sigma, dtype=np.float64)
    n_short = sig.size
    m_long = int(round(n_short * c))
    w = 1.0
    k = 0
    for _ in range(iters):
        edge = lam_plus(m_long, n_short, w)
        k = int(np.searchsorted(-sig, -edge))  # sig descending: count sigma > edge
        if k >= 0.9 * n_short:
            return 0.0, k, False
        w_new = 1.0 - float(np.sum(sig[:k] ** 2))
        if abs(w_new - w) < 1e-6:
            w = w_new
            break
        w = max(w_new, 1e-6)
    return w, k, w > 1e-4


# -- BGN asymptotic formulas (noise G/sqrt(n), G iid N(0,1), c = m/n >= 1) ---


def bgn_sigma_obs(theta, c: float):
    """Observed singular value for spike strength ``theta`` (``theta > c^0.25``)."""
    theta = np.asarray(theta, dtype=np.float64)
    return np.sqrt((1.0 + theta**2) * (c + theta**2)) / theta


def bgn_overlaps(theta, c: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Squared singular-vector overlaps ``(cos^2 theta_u, cos^2 theta_v)``.

    ``theta_u`` is the left (long side m) and ``theta_v`` the right (short
    side n) singular-vector overlap with the true spike direction.
    """
    theta = np.asarray(theta, dtype=np.float64)
    t4 = theta**4
    return (t4 - c) / (t4 + c * theta**2), (t4 - c) / (t4 + theta**2)


def theta_hat(sigma_scaled, c: float):
    """Invert :func:`bgn_sigma_obs`; NaN inside the bulk."""
    s = np.asarray(sigma_scaled, dtype=np.float64)
    y = s**2
    disc = (y - 1.0 - c) ** 2 - 4.0 * c
    th2 = np.where(disc >= 0, (y - 1.0 - c + np.sqrt(np.maximum(disc, 0))) / 2, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(th2)


# -- group-level diagnostic --------------------------------------------------


@dataclass
class MpMatrixDiag:
    """MP peel result for one matrix."""

    w: float  # estimated noise energy share
    k_spike: int  # number of singular values above lam_plus(w)
    lam: float  # lam_plus(m, n, w)
    ok: bool


@dataclass
class MpGroupSummary:
    """MP diagnostic summary for one shape group."""

    n_fit: int  # matrices where peeling converged
    n_total: int
    lam_median: float
    lam_p25: float
    lam_p75: float
    w_median: float
    k_spike_median: float
    ks_D: float  # KS distance of the pooled bulk to MP_c (NaN if no fit)
    ks_n: int  # pooled bulk sample size
    ks_crit_5pct: float  # 1.36 / sqrt(ks_n)

    @property
    def mp_consistent(self) -> bool | None:
        """True if the pooled bulk shape is consistent with MP (D < 2 * crit)."""
        if self.ks_n == 0 or not np.isfinite(self.ks_D):
            return None
        return bool(self.ks_D < 2 * self.ks_crit_5pct)


def diagnose_matrix(est: MatrixEstimate) -> MpMatrixDiag:
    """Run spike peeling + lam_plus on one (F-normalized) matrix estimate."""
    m, n = est.shape
    c = max(m, n) / min(m, n)
    w, k, ok = peel_w_c(est.sigma, c)
    return MpMatrixDiag(w=w, k_spike=k, lam=lam_plus(m, n, w), ok=bool(ok and est.ok))


def ks_shape_check(
    spectra: Sequence[NDArray[np.floating]],
    ws: Sequence[float],
    c: float,
    n_short: int,
) -> tuple[float, int]:
    """KS test of the pooled bulk against the MP_c law.

    For each spectrum the bulk singular values (``sigma <= lam_plus(w)``) are
    mapped to ``y = sigma^2 * n_short / w``, pooled, and compared to the
    MP_c CDF by Kolmogorov-Smirnov distance. Returns ``(D, n_bulk)``.
    """
    ys = []
    for S, w in zip(spectra, ws):
        m_long = int(round(n_short * c))
        edge = lam_plus(m_long, n_short, w)
        bulk = np.asarray(S, dtype=np.float64)
        bulk = bulk[bulk <= edge]
        ys.append(bulk**2 * n_short / w)
    if not ys:
        return float("nan"), 0
    y = np.sort(np.concatenate(ys))
    if y.size == 0:
        return float("nan"), 0
    b = (1.0 + 1.0 / np.sqrt(c)) ** 2
    a = (1.0 - 1.0 / np.sqrt(c)) ** 2
    # Quadratic grid concentrated at the lower edge: for square matrices
    # (a == 0) the MP density diverges like 1/sqrt(y) at 0 and a uniform grid
    # with trapezoid integration badly misestimates the CDF there. With
    # y = a + (hi - a) u^2 the integrand pdf * dy/du is smooth even at a == 0.
    hi = b * 1.001
    u = np.linspace(0.0, 1.0, 4000)
    grid = a + (hi - a) * u**2
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf = np.sqrt(np.maximum(grid - a, 0) * np.maximum(b - grid, 0)) / (
            2 * np.pi * grid / c
        )
    mass = np.nan_to_num(pdf * 2.0 * (hi - a) * u, nan=0.0, posinf=0.0)
    cdf_t = np.concatenate([[0.0], np.cumsum((mass[1:] + mass[:-1]) / 2 * np.diff(u))])
    cdf_t /= max(cdf_t[-1], 1e-12)
    t_at_y = np.interp(y, grid, cdf_t)
    e_at_y = (np.arange(y.size) + 0.5) / y.size
    return float(np.max(np.abs(t_at_y - e_at_y))), int(y.size)


def summarize_group(estimates: Sequence[MatrixEstimate], shape: tuple[int, int]) -> MpGroupSummary:
    """Pool the MP diagnostic over one shape group of matrix estimates."""
    m, n = shape
    c = max(m, n) / min(m, n)
    n_short = min(m, n)
    diags = [diagnose_matrix(e) for e in estimates]
    fits = [(e, d) for e, d in zip(estimates, diags) if d.ok]
    nan = float("nan")
    if not fits:
        return MpGroupSummary(
            n_fit=0, n_total=len(estimates), lam_median=nan, lam_p25=nan,
            lam_p75=nan, w_median=nan, k_spike_median=nan, ks_D=nan, ks_n=0,
            ks_crit_5pct=nan,
        )
    lams = np.array([d.lam for _, d in fits])
    D, n_bulk = ks_shape_check([e.sigma for e, _ in fits], [d.w for _, d in fits], c, n_short)
    return MpGroupSummary(
        n_fit=len(fits),
        n_total=len(estimates),
        lam_median=float(np.median(lams)),
        lam_p25=float(np.percentile(lams, 25)),
        lam_p75=float(np.percentile(lams, 75)),
        w_median=float(np.median([d.w for _, d in fits])),
        k_spike_median=float(np.median([d.k_spike for _, d in fits])),
        ks_D=D,
        ks_n=n_bulk,
        ks_crit_5pct=float(1.36 / np.sqrt(n_bulk)) if n_bulk else nan,
    )
