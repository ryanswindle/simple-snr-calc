"""Search-rate overlay: senpai on-sky measurements + simple-snr-calc model.

senpai's ``search_rate`` plot shows, per catalog magnitude, the sky area/hour
surveyable while still reaching its target sigma -- derived from *measured*
stellar SNRs. This module redraws that on-sky panel from ``plot_data.json`` and
overlays the model's predicted search rate computed under the same night's
measured conditions (see :mod:`.conditions`), so model and reality share axes.
"""

from __future__ import annotations

import json
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


def load_plot_data(path: str | Path) -> dict:
    """Load a senpai ``plot_data.json`` (plain JSON; senpai not required)."""
    with open(path) as f:
        return json.load(f)


def save_overlay_variants(fig, ax, output_path: str | Path) -> list[Path]:
    """Save the titled figure and a title-less ``*_clean`` twin; return both paths.

    Two PNGs are written from one figure (no re-rendering): the titled
    ``output_path`` and, with the in-figure title stripped, a
    ``<stem>_clean<suffix>`` sibling. The clean variant is the paper-ready one --
    its per-night condition summary belongs in the manuscript caption, not baked
    into the image. The titled path is returned first.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    clean_path = output_path.with_name(
        f"{output_path.stem}_clean{output_path.suffix}")
    ax.set_title("")
    fig.savefig(clean_path, dpi=150, bbox_inches="tight")
    return [output_path, clean_path]


def _exposure_to_reach(calc: SNRCalculator, mv: float, target_snr: float,
                       t_lo: float, t_hi: float) -> float | None:
    """Smallest exposure in ``[t_lo, t_hi]`` whose SNR(mv) >= target, else None.

    SNR rises monotonically with exposure for unsaturated sources, so a geometric
    bisection converges; this avoids the coarse fixed grid in ``SNRCalculator.sweep``
    and stays accurate across many decades of exposure.
    """
    if calc.compute_snr(mv, t_lo).snr >= target_snr:
        return t_lo
    if calc.compute_snr(mv, t_hi).snr < target_snr:
        return None
    lo, hi = t_lo, t_hi
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if calc.compute_snr(mv, mid).snr >= target_snr:
            hi = mid
        else:
            lo = mid
    return hi


def model_search_rate_vs_mag(
    config: SNRConfig, mv_step: float = 0.1, max_exposure_s: float = 10.0,
    zero_point_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Model search rate vs magnitude, returning ``(mvs, rates_deg2_per_hr, info)``.

    For each magnitude the model finds, by root-finding (not a fixed exposure
    grid), the exposure ``t_cross`` needed to reach the target sigma; the search
    rate is ``num_fields * fov / (t_cross * num_frames + duty)`` per hour, with
    ``duty = max(step_settle, readout)``. A magnitude whose target sigma is
    unreachable within ``max_exposure_s`` gets rate 0. Raising ``max_exposure_s``
    pushes the rate smoothly toward 0 at faint magnitudes instead of cliff-dropping.

    ``zero_point_offset`` shifts the system zero point (mag); pass
    :data:`GAIA_G_MINUS_V_SUN` to evaluate the native-V model on a Gaia-G axis.
    """
    cfg = config.model_copy(deep=True)
    calc = SNRCalculator(cfg)
    calc.zero_point += zero_point_offset
    obs = cfg.observation
    target = obs.snr_threshold
    t_lo = obs.exposure_range[0]

    mvs = np.arange(obs.mv_range[0], obs.mv_range[1] + mv_step / 2, mv_step)
    duty = max(cfg.optics.step_settle_time,
               1.0 / cfg.detector.frame_rate if cfg.detector.frame_rate else 0.0)
    fov_sq = float(calc.fov[0] * calc.fov[1])

    rates = np.zeros(len(mvs))
    max_tcross = 0.0
    for i, mv in enumerate(mvs):
        t_cross = _exposure_to_reach(calc, float(mv), target, t_lo, max_exposure_s)
        if t_cross is None:
            continue
        cadence_s = t_cross * obs.num_frames + duty
        rates[i] = obs.num_fields * fov_sq / (cadence_s / 3600.0)
        max_tcross = max(max_tcross, t_cross)

    info = {
        "fov_deg": (float(calc.fov[0]), float(calc.fov[1])),
        "fov_sq_deg": fov_sq,
        "fwhm_arcsec": float(calc.fwhm_arcsec),
        "zero_point": float(calc.zero_point),
        "snr_threshold": float(target),
        "step_settle_time": float(cfg.optics.step_settle_time),
        "ensquared_energy": float(calc.ensquared_energy),
        "max_exposure_s": float(max_exposure_s),
        "max_tcross_s": float(max_tcross),
    }
    return mvs, rates, info


def _fmt_exposure(seconds: float) -> str:
    """Human-readable exposure: seconds, or hours when very long."""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"


def _condition_summary(cfg: SNRConfig, cond: NightConditions, info: dict) -> str:
    """One-line description of the measured conditions fed to the model."""
    bits = []
    if cond.zenith_transmission is not None:
        bits.append(f"T$_{{zen}}$={cond.zenith_transmission:.2f}")
    if cond.sky_mag_arcsec2 is not None:
        bits.append(f"sky={cond.sky_mag_arcsec2:.1f} mag/arcsec²")
    if cond.fwhm_px is not None:
        bits.append(f"FWHM={info['fwhm_arcsec']:.1f}\"")
    bits.append(f"FoV={info['fov_sq_deg']:.2f} deg²")
    bits.append(f"overhead={info['step_settle_time']:.1f}s")
    return ", ".join(bits)


def plot_search_rate_overlay(
    plot_data: dict | str | Path,
    conditions: NightConditions | str | Path,
    base_config: str | Path | SNRConfig,
    output_path: str | Path,
    *,
    mv_step: float = 0.1,
    max_exposure_s: float | None = None,
    plt=None,
):
    """Render the search-rate overlay and return the written paths.

    Two PNGs are written: the titled ``output_path`` and a title-less
    ``*_clean`` twin for paper figures (see :func:`save_overlay_variants`);
    both :class:`Path` objects are returned, titled first.

    Parameters
    ----------
    plot_data : senpai ``plot_data.json`` path or already-loaded dict.
    conditions : this night's measured conditions -- a :class:`NightConditions`
        or a path to its ``calibration/night_calibration.json``.
    base_config : simple-snr-calc config (e.g. ``configs/dao.yaml``) path/object;
        its design values are overridden per-night by the measured conditions.
    output_path : where to write the PNG.
    max_exposure_s : max exposure the model may integrate to. Default ``None``
        auto-selects the exposure that just reaches the faintest on-sky magnitude
        (the x-axis edge), so the model curve declines smoothly to ~0 there with
        no cliff -- note this can be very long (tens of thousands of seconds) for
        faint field stars. Pass a number (e.g. 10) to cap it at a realistic value.
    """
    if not isinstance(plot_data, dict):
        plot_data = load_plot_data(plot_data)
    cond = (conditions if isinstance(conditions, NightConditions)
            else load_night_conditions(conditions))
    base = base_config if isinstance(base_config, SNRConfig) else load_config(base_config)

    meta = plot_data.get("meta", {})
    night_id = meta.get("night_id", "")
    d = plot_data.get("plots", {}).get("search_rate")
    if d is None:
        raise ValueError(f"no 'search_rate' plot in plot_data for {night_id!r}")

    if cond.moon_illumination is None and meta.get("moon_illumination") is not None:
        cond.moon_illumination = float(meta["moon_illumination"])

    # On-sky scatter extent sets the magnitude range the model is evaluated over;
    # senpai's fitted grid cadence and target sigma make the model's duty cycle
    # and detection threshold match the measured panel.
    mags = np.asarray(d["mags"], dtype=float)
    mv_range = (math.floor(np.nanmin(mags)), math.ceil(np.nanmax(mags)))
    cfg = apply_conditions(
        base, cond,
        overhead_s=d.get("overhead_s"),
        target_snr=d.get("target_snr"),
        mv_range=mv_range,
    )
    # Auto mode: pick the exposure that just reaches the faintest on-sky
    # magnitude, so the model curve runs down to ~0 at the x-axis edge.
    if max_exposure_s is None:
        calc = SNRCalculator(cfg)
        calc.zero_point += GAIA_G_MINUS_V_SUN  # mv_range is Gaia G; match it
        t_edge = _exposure_to_reach(
            calc, float(mv_range[1]), cfg.observation.snr_threshold,
            cfg.observation.exposure_range[0], 1.0e6)
        max_exposure_s = t_edge if t_edge is not None else 1.0e6
    model_mvs, model_rates, info = model_search_rate_vs_mag(
        cfg, mv_step=mv_step, max_exposure_s=max_exposure_s,
        zero_point_offset=GAIA_G_MINUS_V_SUN)

    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- On-sky data (replicates senpai's _render_search_rate) ---------------
    ax.scatter(d["mags"], d["rates"], alpha=0.3, s=10, color="lightgray",
               label="Individual stars (on-sky)")
    b = d["binned"]
    if b["x"]:
        ax.errorbar(b["x"], b["y"], yerr=[b["err_lo"], b["err_hi"]], fmt="o",
                    color="black", markersize=7, capsize=4, capthick=1.5,
                    elinewidth=1.5, alpha=0.85,
                    label="Binned on-sky (median ± 1σ percentiles)")
    if d.get("median_lim50") is not None:
        ax.axvline(d["median_lim50"], color="firebrick", linestyle="--",
                   linewidth=1.5, alpha=0.8,
                   label=f"median lim. mag (50%) = {d['median_lim50']:.1f}")

    # --- Model curve, grounded on this night's measured conditions -----------
    # Rate declines smoothly toward 0 at the faint end (root-found per magnitude),
    # spanning the full x-axis; any unreachable faint tail is drawn as explicit 0.
    nz = np.nonzero(model_rates > 0)[0]
    i0 = int(nz[0]) if len(nz) else 0
    ax.plot(model_mvs[i0:], model_rates[i0:], "-", color="tab:blue", lw=2.2,
            alpha=0.9,
            label=(f"model (measured cond., "
                   f"≤{_fmt_exposure(info['max_exposure_s'])} exp)"))

    target_snr = d.get("target_snr")
    ax.set_xlabel("Gaia G magnitude")
    ax.set_ylabel(
        f"Search Rate (deg²/hour to {target_snr:.0f}σ)"
        if target_snr else "Search Rate (deg²/hour)"
    )
    ax.set_title(
        f"{night_id}: search rate vs magnitude — on-sky vs model\n"
        f"model: {_condition_summary(cfg, cond, info)}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    paths = save_overlay_variants(fig, ax, output_path)
    plt.close(fig)
    return paths
