"""Ground-truth validation on synthetic spiked spectra (numpy + pytest only).

Synthetic data is exactly the spiked-model regime the theory describes:
momentum ``X = S + E`` with ``S`` a sum of rank-1 spikes of known orientation
and strength, ``E`` iid Gaussian entry noise with Frobenius energy share ``w``
(exactly, by rescaling). Micro-batch gradients are ``G_a = S + F_a`` with fresh
independent noise ``F_a`` of the same scale — shared signal stacks, fresh noise
averages to zero.

All "truth" values are computed in the F-normalized units the estimator works
in, using the actual Frobenius norm of the constructed matrix:

- BGN units: noise entries ~ N(0, 1/n_short); a planted spike of F-normalized
  strength ``theta'`` has BGN strength ``theta = theta' / sqrt(w / m_long)``
  (the Frobenius norm cancels in this ratio).
- observed singular value: ``sigma_obs = sqrt(w / m_long) / fro * so(theta, c)``
  with ``so`` the BGN observed-sigma formula.
- predicted persistence: ``rho = theta'^2 * cos^2_u * cos^2_v / sigma_obs^2``
  in normalized units, i.e. ``theta^2 * pu * pv / so^2`` (scale-free).
- true MP bulk edge: ``lam = (sqrt(w) / fro) * (1/sqrt(m) + 1/sqrt(n))``.
"""

import json

import numpy as np
import pytest

from kneescope import (
    Snapshot,
    SnapshotFormatError,
    SnapshotWriter,
    analyze,
    analyze_matrix,
    load_snapshot,
)
from kneescope.mp_diag import bgn_overlaps, bgn_sigma_obs, lam_plus, peel_w_c

N_MB = 4


def _ortho(rng, dim, k):
    q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return q


def make_group(rng, m, n, spike_strengths, w, n_mats, n_mb=N_MB):
    """Build (momenta, grads, truths) for a spiked shape group.

    spike_strengths are the singular values of the signal S (raw units; the
    matrices are ~unit-norm so they are nearly the F-normalized strengths).
    E and each F_a are scaled to exact Frobenius energy w.
    """
    spikes = np.asarray(spike_strengths, dtype=np.float64)
    r = spikes.size
    m_long, n_short = max(m, n), min(m, n)
    c = m_long / n_short
    moms, grads, truths = [], [], []
    for _ in range(n_mats):
        U = _ortho(rng, m, r)
        V = _ortho(rng, n, r)
        S = (U * spikes) @ V.T
        E = rng.standard_normal((m, n))
        E *= np.sqrt(w) / np.linalg.norm(E)
        X = S + E
        fro = np.linalg.norm(X)
        gs = []
        for _a in range(n_mb):
            F = rng.standard_normal((m, n))
            F *= np.sqrt(w) / np.linalg.norm(F)
            gs.append(S + F)
        moms.append(X)
        grads.append(gs)
        theta = spikes / np.sqrt(w / m_long)
        so = bgn_sigma_obs(theta, c)
        pu, pv = bgn_overlaps(theta, c)
        s0 = np.sqrt(w / m_long) / fro
        truths.append(
            {
                "c": c,
                "theta": theta,
                "sigma_obs": s0 * so,
                "rho_pred": theta**2 * pu * pv / so**2,
                "lam_true": (np.sqrt(w) / fro) * (1.0 / np.sqrt(m) + 1.0 / np.sqrt(n)),
            }
        )
    return moms, grads, truths


def to_snapshot(step, moms, grads, prefix="layers"):
    """Pack per-matrix data into an in-memory Snapshot (4 micro-batches)."""
    paths = [f"{prefix}.{i}.weight" for i in range(len(moms))]
    n_mb = len(grads[0])
    return Snapshot(
        momentum={step: dict(zip(paths, moms))},
        grads={
            step: [{p: grads[i][a] for i, p in enumerate(paths)} for a in range(n_mb)]
        },
    )


# ---------------------------------------------------------------------------
# bulk directions ~ pure noise, spike directions ~ BGN prediction, MP works
# ---------------------------------------------------------------------------


def test_spiked_group_bulk_spikes_and_mp():
    rng = np.random.default_rng(20260825)
    m, n, w = 640, 384, 0.39
    spikes = [0.6, 0.4, 0.3]  # all far above the bulk edge (~0.057)
    moms, grads, truths = make_group(rng, m, n, spikes, w, n_mats=1)
    truth = truths[0]
    est = analyze_matrix(moms[0], grads[0])
    assert est.ok

    r = len(spikes)
    # spike singular directions are the top-r (smallest spike ~0.30 >> bulk ~0.06)
    assert est.sigma[:r] == pytest.approx(truth["sigma_obs"], rel=0.05)

    # bulk directions have rho ~ 0
    assert np.median(np.abs(est.rho[r:])) < 0.1

    # spike directions have rho within the BGN theoretical range
    for i in range(r):
        assert 0.0 < est.rho[i] <= 1.2
        assert est.rho[i] == pytest.approx(truth["rho_pred"][i], abs=0.08)

    # MP diagnostic recovers lambda+ within ~3% (synthetic IS the MP regime)
    w_hat, k_spike, ok = peel_w_c(est.sigma, truth["c"])
    assert ok
    # the 3 planted spikes plus at most edge-fluctuating bulk directions
    assert r <= k_spike <= r + 2
    lam_hat = lam_plus(m, n, w_hat)
    assert lam_hat == pytest.approx(truth["lam_true"], rel=0.03)


# ---------------------------------------------------------------------------
# the detected band brackets the planted signal edge
# ---------------------------------------------------------------------------


def test_band_brackets_planted_signal_edge():
    rng = np.random.default_rng(770031)
    m, n, w = 1024, 256, 0.0455  # bulk edge ~0.025, below the 0.03 bin edge
    # Planted signal strengths: persistence rho climbs with sigma (BGN), so the
    # 0.3-crossing and the 0.9-crossing land in different bins.
    spikes = [0.04, 0.05, 0.065, 0.085, 0.12, 0.16, 0.22, 0.29, 0.38, 0.5]
    moms, grads, truths = make_group(rng, m, n, spikes, w, n_mats=4)
    s_min = min(float(np.min(t["sigma_obs"])) for t in truths)

    result = analyze(to_snapshot(300, moms, grads))
    rep = result.groups[(300, (m, n))]

    assert rep.sigma_star is not None, "low band edge must be found"
    assert rep.sigma_star_upper is not None, "upper band edge must be found"
    # band brackets the true smallest spike sigma (loose, robust tolerances)
    assert rep.sigma_star < 0.95 * s_min
    assert rep.sigma_star_upper >= s_min
    assert rep.sigma_star <= rep.sigma_star_upper
    # single-knee recommendation is the UPPER edge, never sigma*
    assert rep.recommended_knee50 == rep.sigma_star_upper
    # isotropic construction -> anisotropy ratios near 1
    assert 0.5 < rep.anisotropy_row < 2.0
    assert 0.5 < rep.anisotropy_col < 2.0


# ---------------------------------------------------------------------------
# honesty: signal persistent at ALL visible sigma -> no fabricated boundary
# ---------------------------------------------------------------------------


def test_no_collapse_in_view_is_reported_honestly():
    rng = np.random.default_rng(555)
    m, n = 640, 384
    # Full-rank signal, no noise bulk: every visible direction persists (rho ~ 1),
    # so the persistence boundary lies below the measurement window.
    sig = np.geomspace(0.9, 0.01, n)
    sig /= np.linalg.norm(sig)  # F-normalized by construction
    U = _ortho(rng, m, n)
    V = _ortho(rng, n, n)
    S = (U * sig) @ V.T
    grads = [S + 1e-8 * rng.standard_normal((m, n)) for _ in range(N_MB)]
    snap = Snapshot(
        momentum={100: {"blocks.0.attn.wq.weight": S}},
        grads={100: [{"blocks.0.attn.wq.weight": g} for g in grads]},
    )
    result = analyze(snap)
    rep = result.groups[(100, (m, n))]
    assert rep.sigma_star is None
    assert rep.sigma_star_upper is None
    assert rep.recommended_knee50 is None
    assert "no collapse in view" in result.text()


# ---------------------------------------------------------------------------
# snapshot format round-trip and validation
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip_and_validation(tmp_path):
    rng = np.random.default_rng(99)
    m, n, w = 192, 128, 0.3
    moms, grads, _ = make_group(rng, m, n, [0.5, 0.3], w, n_mats=2)
    paths = [f"blocks.{i}.attn.wq.weight" for i in range(2)]

    writer = SnapshotWriter(tmp_path, notes={"run": "test"})
    writer.write_momentum(7, dict(zip(paths, moms)))
    for a in range(N_MB):
        writer.write_grad_microbatch(7, a, {p: grads[i][a] for i, p in enumerate(paths)})

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["steps"] == [7]
    assert manifest["n_microbatches"] == N_MB
    assert manifest["notes"] == {"run": "test"}

    snap = load_snapshot(tmp_path)
    assert snap.steps == [7]
    assert snap.n_microbatches(7) == N_MB
    assert set(snap.momentum[7]) == set(paths)

    result = analyze(snap)
    rep = result.groups[(7, (m, n))]
    assert rep.n_matrices == 2
    assert rep.n_directions == 2 * min(m, n)

    # manifest validation: unsupported version must raise, not guess
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps({"format_version": 2}))
    with pytest.raises(SnapshotFormatError):
        load_snapshot(bad)
    with pytest.raises(SnapshotFormatError):
        load_snapshot(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# structured subsampling of stacked tensors
# ---------------------------------------------------------------------------


def test_split_stacked_subsamples_deterministically():
    arr = np.arange(64 * 8 * 16, dtype=np.float64).reshape(64, 8, 16)
    out = SnapshotWriter.split_stacked("blocks.0.mlp.experts.gu", arr, max_slices=6)
    assert len(out) == 6
    assert all(v.shape == (8, 16) for v in out.values())
    # evenly spaced, includes both ends, deterministic (momentum/grads must match)
    assert sorted(out) == [
        "blocks.0.mlp.experts.gu[e0]",
        "blocks.0.mlp.experts.gu[e12]",
        "blocks.0.mlp.experts.gu[e25]",
        "blocks.0.mlp.experts.gu[e37]",
        "blocks.0.mlp.experts.gu[e50]",
        "blocks.0.mlp.experts.gu[e63]",
    ]
    np.testing.assert_array_equal(out["blocks.0.mlp.experts.gu[e63]"], arr[63])
    # 2D input passes through under the bare path
    two_d = np.zeros((8, 16))
    assert SnapshotWriter.split_stacked("w", two_d)["w"] is two_d


def test_torch_adapter_module_imports_without_torch():
    # The base install is numpy-only; the adapter module must still import.
    import kneescope.capture.torch as cap

    assert hasattr(cap, "TorchKneeProbe")
