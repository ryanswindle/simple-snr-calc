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
from .conditions import (
    GAIA_G_MINUS_V_SUN,
    NightConditions,
    apply_conditions,
    load_night_conditions,
)
from .search_rate import load_plot_data, save_overlay_variants


def model_snr_vs_exposure(
    config: SNRConfig, mags, exp_min: float, exp_max: float, n: int = 60,
    zero_point_offset: float = 0.0,
) -> tuple[np.ndarray, dict, dict]:
    """Model SNR vs exposure for each magnitude.

    Returns ``(exposures, {mag: snr_array}, info)``. One ``SNRCalculator`` is
    built and reused across all magnitudes and exposures. ``zero_point_offset``
    shifts the system zero point (mag); pass :data:`GAIA_G_MINUS_V_SUN` so the
    Gaia-G bin magnitudes are evaluated on the native-V model consistently.
    """
    calc = SNRCalculator(config)
    calc.zero_point += zero_point_offset
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
    conditions: NightConditions | str | Path,
    base_config: str | Path | SNRConfig,
    output_path: str | Path,
    *,
    plt=None,
):
    """Render the SNR-vs-exposure (by-magnitude) overlay; return the written paths.

    Writes the titled ``output_path`` and a title-less ``*_clean`` twin for paper
    figures, returning both :class:`Path` objects (titled first).

    ``conditions`` is this night's measured conditions -- a
    :class:`NightConditions` or a path to its
    ``calibration/night_calibration.json``.
    """
    if not isinstance(plot_data, dict):
        plot_data = load_plot_data(plot_data)
    cond = (conditions if isinstance(conditions, NightConditions)
            else load_night_conditions(conditions))
    base = base_config if isinstance(base_config, SNRConfig) else load_config(base_config)

    meta = plot_data.get("meta", {})
    night_id = meta.get("night_id", "")
    d = plot_data.get("plots", {}).get("snr_vs_exposure")
    if d is None:
        raise ValueError(f"no 'snr_vs_exposure' plot in plot_data for {night_id!r}")
    pooled = d.get("pooled") or []
    if not pooled:
        raise ValueError(f"no pooled SNR-vs-exposure series for {night_id!r}")

    std_exps = d.get("std_exps") or sorted({x for s in pooled for x in s["x"]})
    bins = d.get("bins") or [s["bin"] for s in pooled]
    # Centre magnitude of each pooled series drives the model curve for that bin.
    bin_mags = {s["bin"]: s["bin"] + 0.5 for s in pooled}

    cfg = apply_conditions(base, cond)
    exp_min, exp_max = float(min(std_exps)), float(max(std_exps))
    model_exps, model_curves, info = model_snr_vs_exposure(
        cfg, sorted(bin_mags.values()), exp_min, exp_max,
        zero_point_offset=GAIA_G_MINUS_V_SUN)

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
    labels.append("model")
    # Widen the x-axis so the legend fits inside on the right without overlapping
    # the (rising) curves; the cleared strip must exceed the legend's width.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.18 * (x1 - x0))
    ax.legend(handles, labels, loc="center right", fontsize=8,
              title=r"m$_G$", ncol=1, framealpha=0.9)

    fig.tight_layout()
    paths = save_overlay_variants(fig, ax, output_path)
    plt.close(fig)
    return paths


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
