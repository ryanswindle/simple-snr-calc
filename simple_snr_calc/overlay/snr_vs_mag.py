"""SNR-vs-magnitude overlay: senpai on-sky measurements + simple-snr-calc model.

senpai's ``snr_vs_mag_weathermasked`` plot shows, per exposure time, the measured
median stellar SNR vs catalog (Gaia G) magnitude -- weather-masked and normalized
to airmass = 1 (zenith). This module redraws that panel from ``plot_data.json``
and overlays the model's predicted SNR(mv, t) under the same night's measured
conditions (zenith transmission, sky brightness, seeing), so the model and the
on-sky data share axes per exposure.

Note the model SNR is the peak-pixel SNR; senpai's is its measured (aperture)
SNR, so a vertical offset between the dashed model and the solid on-sky curves is
expected and is itself the model-vs-data comparison -- not corrected away here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..config import SNRConfig, load_config
from ..snr import SNRCalculator
from .conditions import NightConditions, apply_conditions, load_nights_summary
from .search_rate import load_plot_data


def model_snr_vs_mag(
    config: SNRConfig, exposures, mv_min: float, mv_max: float,
    mv_step: float = 0.1,
) -> tuple[np.ndarray, dict, dict]:
    """Model SNR vs magnitude for each exposure.

    Returns ``(mvs, {exp: snr_array}, info)``. One ``SNRCalculator`` is built and
    reused across all magnitudes and exposures.
    """
    calc = SNRCalculator(config)
    mvs = np.arange(mv_min, mv_max + mv_step / 2, mv_step)
    curves: dict = {}
    for t in exposures:
        curves[t] = np.array([calc.compute_snr(float(mv), float(t)).snr
                              for mv in mvs])
    info = {
        "fov_sq_deg": float(calc.fov[0] * calc.fov[1]),
        "fwhm_arcsec": float(calc.fwhm_arcsec),
        "zero_point": float(calc.zero_point),
    }
    return mvs, curves, info


def plot_snr_vs_mag_overlay(
    plot_data: dict | str | Path,
    nights_summary: str | Path | dict[str, NightConditions],
    base_config: str | Path | SNRConfig,
    output_path: str | Path,
    *,
    mv_step: float = 0.1,
    plt=None,
):
    """Render the SNR-vs-magnitude overlay PNG and return its :class:`Path`."""
    if not isinstance(plot_data, dict):
        plot_data = load_plot_data(plot_data)
    if not isinstance(nights_summary, dict):
        nights_summary = load_nights_summary(nights_summary)
    base = base_config if isinstance(base_config, SNRConfig) else load_config(base_config)

    meta = plot_data.get("meta", {})
    night_id = meta.get("night_id", "")
    d = plot_data.get("plots", {}).get("snr_vs_mag_weathermasked")
    if d is None:
        raise ValueError(
            f"no 'snr_vs_mag_weathermasked' plot in plot_data for {night_id!r}")
    if not d.get("lines"):
        raise ValueError(f"no on-sky SNR-vs-mag lines for {night_id!r}")

    cond = nights_summary.get(night_id)
    if cond is None:
        raise KeyError(
            f"night {night_id!r} not in nights_summary ({sorted(nights_summary)})")

    # Exposures and magnitude extent come straight from the on-sky panel so the
    # model is drawn over exactly the same curves.
    exposures = [ln["exp"] for ln in d["lines"]]
    all_x = [x for ln in d["lines"] for x in ln["x"]]
    mv_min, mv_max = math.floor(min(all_x)), math.ceil(max(all_x))

    cfg = apply_conditions(base, cond, mv_range=(mv_min, mv_max))
    model_mvs, model_curves, info = model_snr_vs_mag(
        cfg, exposures, mv_min, mv_max, mv_step=mv_step)

    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    std_exps = d.get("std_exps") or sorted(exposures)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(std_exps), 1)))
    color_of = {t: cmap[i] for i, t in enumerate(std_exps)}

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- On-sky data (replicates senpai's _render_snr_vs_mag_weathermasked) ---
    for ln in d["lines"]:
        ax.plot(ln["x"], ln["y"], "o-", color=color_of.get(ln["exp"], "gray"),
                ms=4, lw=1.5, alpha=0.85, label=f"{int(ln['exp'])}s")
    lim = d.get("lim50")
    if lim is not None:
        ax.axvspan(lim["lo"], lim["hi"], color="red", alpha=0.10)
        ax.axvline(lim["lo"], color="red", ls=":", lw=1, alpha=0.6)
        ax.axvline(lim["hi"], color="red", ls=":", lw=1, alpha=0.6)
        ax.axvline(lim["med"], color="red", ls="--", lw=1.8,
                   label=f"lim50 = {lim['med']:.2f} "
                         f"(16/84: {lim['lo']:.2f}–{lim['hi']:.2f})")
    if d.get("min_meas_snr") is not None:
        ax.axhline(d["min_meas_snr"], color="gray", ls=":", lw=1,
                   label=f"SNR = {d['min_meas_snr']:.0f}")

    # --- Model curves (dashed, same color per exposure) ----------------------
    for t in exposures:
        ax.plot(model_mvs, model_curves[t], "--", color=color_of.get(t, "gray"),
                lw=1.8, alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("Gaia G magnitude")
    ax.set_ylabel("SNR (normalized to airmass = 1)")
    zp_txt = (f"ZP {d['zp_mode']:.2f}±{d['zp_sig']:.2f}"
              if d.get("zp_mode") is not None else "")
    ax.set_title(
        f"{night_id}: SNR vs magnitude — on-sky vs model\n"
        f"on-sky weather-masked {zp_txt}; "
        f"model (dashed): {_cond_line(cond, info)}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, which="both")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", ls="--", lw=1.8))
    labels.append("simple-snr-calc model")
    ax.legend(handles, labels, loc="upper right", fontsize=8, title="exposure")

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
