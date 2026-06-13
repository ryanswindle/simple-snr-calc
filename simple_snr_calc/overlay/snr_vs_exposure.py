"""SNR-vs-exposure overlay: senpai on-sky measurements + simple-snr-calc model.

senpai's ``snr_vs_exposure`` plot (the by-magnitude / pooled panel) shows, per
catalog-magnitude bin, the measured median stellar SNR vs exposure time --
weather-masked, airmass-normalized, pooled over coverage + photometric tasks.
This module redraws that panel from ``plot_data.json`` and overlays the model's
predicted SNR(mv, t) vs exposure at each bin's centre magnitude, under the same
night's measured conditions.

As with the SNR-vs-magnitude overlay, the model SNR is peak-pixel while senpai's
is its measured (aperture) SNR, so a vertical offset between the dashed model and
the solid on-sky series is expected and is itself the comparison.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..config import SNRConfig, load_config
from ..snr import SNRCalculator
from .conditions import NightConditions, apply_conditions, load_nights_summary
from .search_rate import load_plot_data


def model_snr_vs_exposure(
    config: SNRConfig, mags, exp_min: float, exp_max: float, n: int = 60,
) -> tuple[np.ndarray, dict, dict]:
    """Model SNR vs exposure for each magnitude.

    Returns ``(exposures, {mag: snr_array}, info)``. One ``SNRCalculator`` is
    built and reused across all magnitudes and exposures.
    """
    calc = SNRCalculator(config)
    exps = np.linspace(exp_min, exp_max, n)
    curves: dict = {}
    for m in mags:
        curves[m] = np.array([calc.compute_snr(float(m), float(t)).snr
                              for t in exps])
    info = {
        "fov_sq_deg": float(calc.fov[0] * calc.fov[1]),
        "fwhm_arcsec": float(calc.fwhm_arcsec),
        "zero_point": float(calc.zero_point),
    }
    return exps, curves, info


def plot_snr_vs_exposure_overlay(
    plot_data: dict | str | Path,
    nights_summary: str | Path | dict[str, NightConditions],
    base_config: str | Path | SNRConfig,
    output_path: str | Path,
    *,
    plt=None,
):
    """Render the SNR-vs-exposure (by-magnitude) overlay PNG; return its Path."""
    if not isinstance(plot_data, dict):
        plot_data = load_plot_data(plot_data)
    if not isinstance(nights_summary, dict):
        nights_summary = load_nights_summary(nights_summary)
    base = base_config if isinstance(base_config, SNRConfig) else load_config(base_config)

    meta = plot_data.get("meta", {})
    night_id = meta.get("night_id", "")
    d = plot_data.get("plots", {}).get("snr_vs_exposure")
    if d is None:
        raise ValueError(f"no 'snr_vs_exposure' plot in plot_data for {night_id!r}")
    pooled = d.get("pooled") or []
    if not pooled:
        raise ValueError(f"no pooled SNR-vs-exposure series for {night_id!r}")

    cond = nights_summary.get(night_id)
    if cond is None:
        raise KeyError(
            f"night {night_id!r} not in nights_summary ({sorted(nights_summary)})")

    std_exps = d.get("std_exps") or sorted({x for s in pooled for x in s["x"]})
    bins = d.get("bins") or [s["bin"] for s in pooled]
    # Centre magnitude of each pooled series drives the model curve for that bin.
    bin_mags = {s["bin"]: s["bin"] + 0.5 for s in pooled}

    cfg = apply_conditions(base, cond)
    exp_min, exp_max = float(min(std_exps)), float(max(std_exps))
    model_exps, model_curves, info = model_snr_vs_exposure(
        cfg, sorted(bin_mags.values()), exp_min, exp_max)

    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(bins), 1)))
    color_of = {lo: colors[i] for i, lo in enumerate(bins)}

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- On-sky data (replicates senpai's pooled by-magnitude panel) ---------
    for ser in pooled:
        ax.errorbar(ser["x"], ser["y"], yerr=[ser["e_lo"], ser["e_hi"]],
                    fmt="o-", color=color_of.get(ser["bin"], "gray"), alpha=0.8,
                    linewidth=1.5, markersize=5, capsize=3,
                    label=f"{ser['bin'] + 0.5:.1f}")

    # --- Model curves (dashed, same colour per magnitude bin) ----------------
    for ser in pooled:
        mv = bin_mags[ser["bin"]]
        ax.plot(model_exps, model_curves[mv], "--",
                color=color_of.get(ser["bin"], "gray"), lw=1.8, alpha=0.7)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(std_exps)
    ax.set_xlabel("Exposure Time [seconds]")
    ax.set_ylabel("SNR (normalized to airmass = 1)")
    ax.set_title(
        f"{night_id}: SNR vs exposure — on-sky vs model "
        f"(coverage+photometric)\nmodel (dashed): {_cond_line(cond, info)}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, which="both")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", ls="--", lw=1.8))
    labels.append("simple-snr-calc model")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, title=r"m$_G$", ncol=1)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _cond_line(cond: NightConditions, info: dict) -> str:
    bits = []
    if cond.zenith_transmission is not None:
        bits.append(f"T$_{{zen}}$={cond.zenith_transmission:.2f}")
    if cond.sky_mag_arcsec2 is not None:
        bits.append(f"sky={cond.sky_mag_arcsec2:.1f}")
    if cond.fwhm_px is not None:
        bits.append(f"FWHM={info['fwhm_arcsec']:.1f}\"")
    bits.append(f"ZP={info['zero_point']:.2f}")
    return ", ".join(bits)
