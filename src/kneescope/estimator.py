"""Core per-matrix persistence estimator (assumption-free).

For one 2D momentum matrix ``M`` (the EMA-of-gradients matrix that is fed into
the orthogonalization: post-EMA/nesterov, pre-normalization) and >= 2
independent micro-batch gradients ``G_a`` from consecutive optimizer updates:

1. F-normalize, ``U = M / ||M||_F`` so ``sum(sigma_k^2) = 1``, and take the SVD,
   giving singular directions ``(u_k, v_k)`` and values ``sigma_k``.
2. Project each micro-batch gradient onto the singular directions,
   ``d[a, k] = u_k^T G_a v_k``, expressed in the same F-normalized units.
3. Cross-microbatch signal energy ``s2_k = mean_{a<b} d[a,k] * d[b,k]``.
   Independent fresh noise averages to zero across pairs; persistent signal
   stacks.
4. Persistence ratio ``rho_k = s2_k / sigma_k^2`` with an estimator noise floor
   ``floor_k = max(Var_a(d_k) - s2_k, 0) / sqrt(n_pairs) / sigma_k^2``.
   (``Var_a(d) = signal^2 + tau^2`` with ``tau`` the fresh-noise projection, so
   ``max(Var - s2, 0)`` estimates ``tau^2``; averaging over ``n_pairs`` pairs
   shrinks it by ``sqrt(n_pairs)``.)

Persistent (signal) directions have ``rho ~ 1``; pure-noise directions
``rho ~ 0``.

The module also computes the noise-anisotropy diagnostic: row/column second
moments of the *centered residual* gradients. The residual must be measured
after removing the shared signal — all micro-batches share the same signal, so
centering on the micro-batch mean leaves pure noise (times the finite-sample
correction ``sqrt(n_mb / (n_mb - 1))``). Measuring raw gradient row variance
instead would misreport the signal's row structure as noise anisotropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EPS = 1e-30
_MIN_FROBENIUS = 1e-20


@dataclass
class MatrixEstimate:
    """Per-direction persistence quantities for one momentum matrix.

    All spectral quantities are in F-normalized units (``sum(sigma**2) == 1``).
    """

    shape: tuple[int, int]
    sigma: NDArray[np.float64]  # singular values, descending
    rho: NDArray[np.float64]  # persistence ratio s2 / sigma^2 per direction
    rho_floor: NDArray[np.float64]  # estimator noise floor, same units as rho
    s2: NDArray[np.float64]  # cross-microbatch signal energy per direction
    row_noise: NDArray[np.float64]  # per-row residual second moments (all rows, all mbs)
    col_noise: NDArray[np.float64]  # per-column residual second moments
    zipf_slope: float  # log-sigma vs log-rank slope, mid-spectrum (heavy-tail signature)
    n_microbatches: int
    ok: bool = True  # False if the matrix was degenerate (near-zero/non-finite norm)

    @property
    def n_directions(self) -> int:
        return int(self.sigma.size)


def analyze_matrix(
    momentum: ArrayLike,
    grads: Sequence[ArrayLike],
) -> MatrixEstimate:
    """Estimate per-direction persistence for one momentum matrix.

    Parameters
    ----------
    momentum:
        2D array, the momentum matrix fed into the orthogonalization
        (post-EMA/nesterov, pre-normalization; raw scale is fine, it is
        F-normalized internally).
    grads:
        Sequence of >= 2 micro-batch gradients, same shape as ``momentum``,
        each at its true per-microbatch scale (not divided by the
        accumulation count). The micro-batches must be independent draws of
        fresh noise around the shared persistent signal.
    """
    M = np.asarray(momentum, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"momentum must be 2D, got shape {M.shape}")
    if len(grads) < 2:
        raise ValueError(
            "at least two independent micro-batch gradients are required "
            f"(got {len(grads)})"
        )
    Gs = [np.asarray(g, dtype=np.float64) for g in grads]
    for a, G in enumerate(Gs):
        if G.shape != M.shape:
            raise ValueError(
                f"grad {a} has shape {G.shape}, expected {M.shape} "
                "(adapters must split stacked tensors into matching 2D matrices)"
            )
    m, n = M.shape
    k = min(m, n)

    fro = np.linalg.norm(M)
    if not np.isfinite(fro) or fro < _MIN_FROBENIUS:
        z = np.zeros(k)
        return MatrixEstimate(
            shape=(m, n), sigma=z, rho=z.copy(), rho_floor=z.copy(),
            s2=z.copy(), row_noise=np.ones(m), col_noise=np.ones(n),
            zipf_slope=float("nan"), n_microbatches=len(Gs), ok=False,
        )

    Mn = M / fro
    U, S, Vt = np.linalg.svd(Mn, full_matrices=False)

    # d[a, k] = u_k^T G_a v_k in F-normalized units.
    # diag(U^T G V) is the same as einsum("mk,mn,kn->k", U, G, Vt), much faster.
    ds = np.stack([np.diag(U.T @ (G / fro) @ Vt.T) for G in Gs])  # (n_mb, k)
    n_mb = ds.shape[0]
    n_pairs = n_mb * (n_mb - 1) // 2

    # Cross-microbatch signal energy: independent fresh noise has zero-mean
    # cross-pair products; persistent signal stacks.
    s2 = np.zeros(k)
    for a in range(n_mb):
        for b in range(a + 1, n_mb):
            s2 += ds[a] * ds[b]
    s2 /= max(n_pairs, 1)

    s2_clip = np.maximum(S**2, _EPS)
    rho = s2 / s2_clip

    # Estimator noise floor: Var_a(d) = signal^2 + tau^2 and s2 -> signal^2,
    # so tau2_hat = max(Var - s2, 0); the pair-average floor is tau2/sqrt(n_pairs).
    tau2 = np.maximum(ds.var(axis=0, ddof=1) - s2, 0.0)
    rho_floor = tau2 / np.sqrt(max(n_pairs, 1)) / s2_clip

    # Noise anisotropy from centered residual gradients. The shared signal cancels
    # in the residuals; sqrt(n_mb / (n_mb - 1)) corrects the finite-sample bias of
    # centering (for n_mb = 4 this is the classic sqrt(4/3)).
    Gcat = np.stack(Gs)  # (n_mb, m, n)
    Gres = (Gcat - Gcat.mean(axis=0, keepdims=True)) * np.sqrt(n_mb / (n_mb - 1))
    row_noise = (Gres**2).mean(axis=2).ravel()
    col_noise = (Gres**2).mean(axis=1).ravel()

    return MatrixEstimate(
        shape=(m, n), sigma=S, rho=rho, rho_floor=rho_floor, s2=s2,
        row_noise=row_noise, col_noise=col_noise, zipf_slope=_zipf_slope(S),
        n_microbatches=n_mb, ok=True,
    )


def _zipf_slope(S: NDArray[np.float64]) -> float:
    """Slope of log sigma vs log rank over the mid-spectrum.

    MP bulk spectra are nearly flat in this coordinate; a slope <= -0.3 is a
    heavy-tailed signal-spectrum signature.
    """
    k = S.size
    k_mid = k // 4
    idx = np.arange(k_mid, max(k_mid + 2, 3 * k // 4))
    if idx.size >= 2 and np.all(S[idx] > 0):
        return float(np.polyfit(np.log(idx + 1), np.log(S[idx]), 1)[0])
    return float("nan")


def anisotropy_ratios(
    row_noise: NDArray[np.float64],
    col_noise: NDArray[np.float64],
    m: int,
    n: int,
) -> tuple[float, float]:
    """Row/column noise-anisotropy ratios for pooled residual second moments.

    Under isotropic iid noise each row second moment is the mean of ``n``
    squared iid entries, so its coefficient of variation squared is ``2 / n``
    (and ``2 / m`` for columns). The ratio rescales the observed CV^2 by that
    null value: 1.0 = isotropic. Values >> 1 mean the iid reading of the
    estimator noise floor is only approximate.
    """
    r_row = float("nan")
    r_col = float("nan")
    if row_noise.size and row_noise.mean() > 0:
        r_row = float((row_noise.var() / row_noise.mean() ** 2) / (2.0 / n))
    if col_noise.size and col_noise.mean() > 0:
        r_col = float((col_noise.var() / col_noise.mean() ** 2) / (2.0 / m))
    return r_row, r_col
