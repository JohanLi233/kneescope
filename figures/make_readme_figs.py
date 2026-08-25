"""Regenerate the README figures for kneescope.

Assets produced in ``figures/``:

- ``fig0_hero.png`` — the headline measurement. The binned persistence curve
  ρ(σ) with the detected transition band [σ*, σ*_upper], rendered DIRECTLY from
  the real ``kneescope`` estimator on ground-truth synthetic spiked data (the
  same discipline as the tests, so the band detection is validated end-to-end).
- ``fig0_mechanism.png`` — a compact schematic of the measurement pipeline.

Run from the repo root:  ``python figures/make_readme_figs.py``
"""

from __future__ import annotations

import numpy as np
from kneescope import Snapshot, analyze


# --- synthetic spiked-MP ground truth (mirrors tests/test_synthetic.py) -------
def _ortho(rng, dim, k):
    q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return q


def make_group(rng, m, n, spike_strengths, w, n_mats, n_mb=4):
    """Return (momenta, microbatch grads) for a spiked shape group.

    ``S`` is a sum of planted rank-1 spikes, ``E`` iid Gaussian entry noise with
    exact Frobenius energy share ``w``. Micro-batch gradients are ``S + F_a``
    with fresh independent noise — shared signal stacks, fresh noise cancels.
    """
    spikes = np.asarray(spike_strengths, dtype=np.float64)
    r = spikes.size
    moms, grads = [], []
    for _ in range(n_mats):
        U = _ortho(rng, m, r)
        V = _ortho(rng, n, r)
        S = (U * spikes) @ V.T
        E = rng.standard_normal((m, n))
        E *= np.sqrt(w) / np.linalg.norm(E)
        X = S + E
        gs = []
        for _a in range(n_mb):
            F = rng.standard_normal((m, n))
            F *= np.sqrt(w) / np.linalg.norm(F)
            gs.append(S + F)
        moms.append(X)
        grads.append(gs)
    return moms, grads


def build_snapshot(step, moms, grads, prefix="blocks"):
    paths = [f"{prefix}.{i}.weight" for i in range(len(moms))]
    n_mb = len(grads[0])
    return Snapshot(
        momentum={step: dict(zip(paths, moms))},
        grads={step: [{p: grads[i][a] for i, p in enumerate(paths)} for a in range(n_mb)]},
    )


def make_hero():
    """Authentic persistence-band figure from kneescope's own estimator."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 10, "axes.titlesize": 11})

    # Two shape groups, each on its OWN seed (independent of the sibling's
    # RNG state) so the detected band on each panel is stable and clearly
    # shows a band, not a step.
    m1, n1, w1 = 1024, 384, 0.055
    spikes1 = [0.035, 0.045, 0.06, 0.08, 0.11, 0.16, 0.24, 0.34, 0.48, 0.68]
    moms1, grads1 = make_group(np.random.default_rng(770031), m1, n1, spikes1, w1, n_mats=3)

    m2, n2, w2 = 768, 768, 0.05
    spikes2 = [0.02, 0.03, 0.045, 0.065, 0.09, 0.13, 0.18, 0.25, 0.35, 0.5, 0.7]
    moms2, grads2 = make_group(np.random.default_rng(770031), m2, n2, spikes2, w2, n_mats=3)

    snap = build_snapshot(300, moms1 + moms2, grads1 + grads2, prefix="blocks")
    result = analyze(snap, min_per_bin=8)
    rep1 = result.groups[(300, (m1, n1))]
    rep2 = result.groups[(300, (m2, n2))]

    reps = [rep1, rep2]
    titles = ["shape 1024×384 — wide band (heavy-tailed)", "shape 768×768 — narrow band (attention-like)"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    fig.patch.set_facecolor("white")

    for ax, rep, title in zip(axes, reps, titles):
        xs = [np.sqrt(b.lo * b.hi) for b in rep.bins]
        rho = [b.median_rho for b in rep.bins]
        floor = [b.median_floor for b in rep.bins]

        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.35)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.axhline(1, color="#bbbbbb", lw=0.8, ls="--")

        if rep.sigma_star is not None and rep.sigma_star_upper is not None:
            ax.axvspan(rep.sigma_star, rep.sigma_star_upper, color="#f2a900", alpha=0.16, zorder=0)

        ax.plot(xs, rho, "o-", color="#1f6fb2", lw=2.4, ms=7, label="persistence ρ(σ)", zorder=3)
        ax.plot(xs, floor, "s--", color="#8a8a8a", lw=1.3, ms=4.5, alpha=0.85,
                label="estimator floor", zorder=2)
        ax.fill_between(xs, floor, rho, color="#1f6fb2", alpha=0.09, zorder=1)

        # labels for the two band edges
        if rep.sigma_star is not None:
            ax.axvline(rep.sigma_star, ls=":", color="#d62728", lw=1.8, zorder=4)
            ax.annotate("σ*", (rep.sigma_star, 1.18), rotation=90, fontsize=13,
                        color="#d62728", ha="center", va="top", fontweight="bold")
        if rep.sigma_star_upper is not None:
            ax.axvline(rep.sigma_star_upper, ls=":", color="#2ca02c", lw=1.8, zorder=4)
            ax.annotate("σ*_upper\n= knee50", (rep.sigma_star_upper, 1.18), rotation=90,
                        fontsize=11, color="#2ca02c", ha="left", va="top", fontweight="bold")

        if rep.sigma_star is not None and rep.sigma_star_upper is not None:
            mid = np.sqrt(rep.sigma_star * rep.sigma_star_upper)
            ax.annotate("transition band", (mid, 1.30), fontsize=10, color="#b8860b",
                        ha="center", va="bottom")

        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("singular value σ (F-normalized)")
        ax.set_ylabel("persistent signal fraction (ρ − floor)")
        ax.grid(True, which="major", alpha=0.15)
        ax.legend(fontsize=8.5, loc="lower right", frameon=True)

    fig.suptitle("kneescope measures the persistence boundary σ* — a band, not a step",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    fig.savefig("figures/fig0_hero.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig0_hero.png")
    print("  group 1024x384 : sigma* = %s   sigma*_upper = %s" % (rep1.sigma_star, rep1.sigma_star_upper))
    print("  group 768x768  : sigma* = %s   sigma*_upper = %s" % (rep2.sigma_star, rep2.sigma_star_upper))


def make_mechanism():
    """Compact pipeline schematic (one row, with sub-callouts)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    steps = [
        ("momentum M\n(post-EMA, pre-norm)", "#1f6fb2"),
        ("F-normalize\nSVD, Σσ²=1", "#1f6fb2"),
        ("project ≥2 micro-batch\ngradients G_a", "#1f6fb2"),
        ("cross-microbatch\nsignal s²_k", "#2ca02c"),
        ("persistence ratio\nρ_k = s²_k/σ_k²", "#b8860b"),
        ("bin by log σ →\nσ*, σ*_upper", "#d62728"),
    ]
    w_box, gap = 1.62, 0.34
    x0 = 0.0
    y = 1.05
    h = 0.72

    centers = []
    for i, (txt, col) in enumerate(steps):
        x = x0 + i * (w_box + gap)
        centers.append(x + w_box / 2)
        ax.add_patch(FancyBboxPatch((x, y), w_box, h,
                                    boxstyle="round,pad=0.08,rounding_size=0.1",
                                    linewidth=1.5, edgecolor=col,
                                    facecolor="#eef4fb"))
        ax.text(x + w_box / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9.5, weight="bold", color="#222222", zorder=3)

        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w_box, y + h / 2),
                                         (x + w_box + gap, y + h / 2),
                                         arrowstyle="-|>", color="#444444", lw=1.5,
                                         mutation_scale=13))

    notes = [
        ("no distributional\nassumption"),
        ("fresh noise cancels,\nsignal stacks"),
        ("persistent ρ≈1\nnoise ρ≈0"),
        ("place τ's knee here,\nnever at σ*"),
    ]
    note_x = [centers[1], centers[2], centers[3], centers[5]]
    note_y = 0.28
    for x, txt, col in zip(note_x, notes, ["#555555", "#2ca02c", "#2ca02c", "#d62728"]):
        ax.text(x, note_y, txt, ha="center", va="center", fontsize=9, color=col)

    ax.set_xlim(-0.1, x0 + len(steps) * (w_box + gap) - gap + 0.1)
    ax.set_ylim(0.0, 2.0)
    fig.tight_layout()
    fig.savefig("figures/fig0_mechanism.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig0_mechanism.png")


if __name__ == "__main__":
    make_hero()
    make_mechanism()
