"""Text and figure rendering for AnalysisResult / BandReport."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bands import AnalysisResult, BandReport, BoundaryStatus

_KNEE_WARNING = (
    "WARNING: do NOT place the knee at sigma* with a binary zero-below/flat-above "
    "rule — long-horizon harm (see the companion paper 'The Persistence "
    "Boundary', sec. 8). Single-knee schedules: place knee50 at sigma*_upper."
)


def _fmt_sig(v: float) -> str:
    return f"{v:g}"


def _fmt_boundary(value: float | None, status: BoundaryStatus, threshold: float) -> str:
    if value is not None:
        return f"{_fmt_sig(value):>12}   (first bin with excess >= {threshold:.2f})"
    return f"{status.value}"


def format_group(rep: BandReport) -> str:
    """Render one shape-group report as text."""
    m, n = rep.shape
    lines = [
        f"step {rep.step} - shape {m}x{n} - {rep.n_matrices} matrices - "
        f"{rep.n_directions} directions",
    ]
    if rep.spectrum:
        s = rep.spectrum
        lines.append(
            "  spectrum (median across matrices): "
            f"sigma1 {s['sigma1']:.3g}  sigma25% {s['sigma25']:.3g}  "
            f"sigma50% {s['sigma50']:.3g}  sigma75% {s['sigma75']:.3g}  "
            f"sigma_min {s['sigma_min']:.3g}  zipf {rep.zipf_median:+.2f}"
        )
    lines.append(
        f"  noise anisotropy (centered residuals): rows {rep.anisotropy_row:.2f}  "
        f"cols {rep.anisotropy_col:.2f}  (1.0 = isotropic; the iid reading of the "
        "floor is approximate away from 1)"
    )
    if rep.bins:
        lines.append(
            f"  {'sigma bin':>26} {'count':>7} {'median rho':>11} "
            f"{'median floor':>13} {'excess':>8}"
        )
        for b in rep.bins:
            lines.append(
                f"  [{_fmt_sig(b.lo):>10}, {_fmt_sig(b.hi):>10}) {b.n:>7} "
                f"{b.median_rho:>11.3f} {b.median_floor:>13.3f} {b.excess:>8.3f}"
            )
    else:
        lines.append("  (no populated bins)")
    lines.append(f"  sigma*       = {_fmt_boundary(rep.sigma_star, rep.star_status, rep.low_threshold)}")
    lines.append(f"  sigma*_upper = {_fmt_boundary(rep.sigma_star_upper, rep.upper_status, rep.high_threshold)}")
    knee = rep.recommended_knee50
    if knee is not None:
        lines.append(
            f"  recommended knee50 = {_fmt_sig(knee)}   <- place the whitening knee "
            "at the UPPER band edge"
        )
    else:
        lines.append("  recommended knee50 = (none: upper band edge not in view)")
    lines.append(f"  {_KNEE_WARNING}")
    if rep.mp is not None:
        mp = rep.mp
        lines.append(
            "  MP diagnostic (assumption-laden; known to fail on heavy-tailed "
            "real spectra):"
        )
        if mp.n_fit:
            verdict = {True: "consistent with MP", False: "deviates from MP", None: "n/a"}[
                mp.mp_consistent
            ]
            lines.append(
                f"    lambda+ med {mp.lam_median:.4g}  IQR [{mp.lam_p25:.4g}, "
                f"{mp.lam_p75:.4g}]  w med {mp.w_median:.4f}  spikes med "
                f"{mp.k_spike_median:g}  fit {mp.n_fit}/{mp.n_total}"
            )
            if mp.ks_n:
                lines.append(
                    f"    bulk shape KS D {mp.ks_D:.4f} (n={mp.ks_n}, 5% crit "
                    f"~{mp.ks_crit_5pct:.4f}) -> {verdict}"
                )
        else:
            lines.append(
                f"    no matrix could be MP-fit (peeling diverged, w -> 0): "
                f"spectrum is not of MP+spikes form (0/{mp.n_total})"
            )
    return "\n".join(lines)


def format_report(result: AnalysisResult) -> str:
    """Render the full multi-group text report."""
    header = [
        "kneescope persistence report",
        "=" * 72,
        f"thresholds: sigma* at excess >= {result.low:.2f}, "
        f"sigma*_upper at excess >= {result.high:.2f}; "
        f"log-sigma bins [{_fmt_sig(result.bin_edges[0])} .. "
        f"{_fmt_sig(result.bin_edges[-1])}], medians within bin",
        "",
    ]
    if result.warnings:
        header.extend(f"note: {w}" for w in result.warnings)
        header.append("")
    if not result.groups:
        header.append("no analyzable shape groups found")
        return "\n".join(header)
    body = [format_group(rep) for rep in result.groups.values()]
    return "\n".join(header) + "\n\n".join(body)


def plot_result(
    result: AnalysisResult,
    path: str | Path | None = None,
    *,
    show: bool = False,
):
    """Plot median rho(sigma) per shape group with the transition band shaded.

    Requires matplotlib (lazy import). Returns the figure.
    """
    import matplotlib.pyplot as plt

    groups = list(result.groups.values())
    if not groups:
        raise ValueError("no shape groups to plot")
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(7.5, 3.4 * len(groups)), squeeze=False
    )
    for ax, rep in zip(axes.ravel(), groups):
        if rep.bins:
            xs = [np.sqrt(b.lo * b.hi) for b in rep.bins]
            ax.plot(xs, [b.median_rho for b in rep.bins], "o-", label="median rho")
            ax.plot(
                xs,
                [b.median_floor for b in rep.bins],
                "s--",
                alpha=0.7,
                label="median floor",
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.15)
        if rep.sigma_star is not None and rep.sigma_star_upper is not None:
            ax.axvspan(
                rep.sigma_star,
                rep.sigma_star_upper,
                color="orange",
                alpha=0.2,
                label="transition band",
            )
        for x, name in (
            (rep.sigma_star, "sigma*"),
            (rep.sigma_star_upper, "sigma*_upper = knee50"),
        ):
            if x is not None:
                ax.axvline(x, ls=":", color="k", alpha=0.6)
        star, upper = rep.sigma_star, rep.sigma_star_upper
        if star is not None and upper is not None and star == upper:
            ax.annotate(
                "sigma* = sigma*_upper = knee50",
                (star, 1.05),
                rotation=90,
                fontsize=8,
                va="top",
            )
        else:
            for x, name in ((star, "sigma*"), (upper, "sigma*_upper = knee50")):
                if x is not None:
                    ax.annotate(name, (x, 1.05), rotation=90, fontsize=8, va="top")
        m, n = rep.shape
        ax.set_title(f"step {rep.step} - {m}x{n} - {rep.n_matrices} matrices")
        ax.set_xlabel("sigma (F-normalized)")
        ax.set_ylabel("persistence rho")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig
