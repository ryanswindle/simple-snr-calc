"""Plotting functions for SNR analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from ._constants import SUN_MV
from .noise import NOISE_SOURCES
from .target import phase_angle_factor

# Color/label for the default combined band and each individual noise source,
# shared across all three panels so a source reads the same everywhere.
_BAND_COLORS = {
    "combined": "#555555",        # grey — the default, headline band
    "target_shot": "#ff7f0e",     # orange
    "sky_background": "#1f77b4",   # blue
    "dark_current": "#2ca02c",     # green
    "read": "#d62728",             # red
    "quantization": "#9467bd",     # purple
}
_BAND_LABELS = {
    "combined": "all noise (1σ)",
    "target_shot": "target shot",
    "sky_background": "sky bkg",
    "dark_current": "dark",
    "read": "read",
    "quantization": "quant.",
}
# Order of the per-source overlay drawn when ``visualize_noise`` is set.
_OVERLAY_SOURCES = tuple(NOISE_SOURCES)
_ALPHA_COMBINED = 0.22
_ALPHA_SOURCE = 0.15


def _fill_band(ax, x, nominal, half_width, color, alpha, *, y_floor):
    """fill_between a ±half_width band around ``nominal``, clamped for log axes."""
    half_width = np.asarray(half_width, dtype=float)
    lo = np.maximum(nominal - half_width, y_floor)
    hi = nominal + half_width
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def _draw_snr_bands(ax, x, nominal, band_unit, sigma, visualize_noise, *,
                    y_floor=1e-3):
    """Shade the default combined SNR band, plus per-source bands if requested.

    ``band_unit`` maps each band key ("combined" and every noise source) to its
    sigma=1 half-width array aligned with ``x``. No-op when ``sigma <= 0``.
    """
    if sigma <= 0:
        return
    nominal = np.asarray(nominal, dtype=float)
    _fill_band(ax, x, nominal, sigma * np.asarray(band_unit["combined"]),
               _BAND_COLORS["combined"], _ALPHA_COMBINED, y_floor=y_floor)
    if visualize_noise:
        for s in _OVERLAY_SOURCES:
            _fill_band(ax, x, nominal, sigma * np.asarray(band_unit[s]),
                       _BAND_COLORS[s], _ALPHA_SOURCE, y_floor=y_floor)


def _band_legend_handles(sigma, visualize_noise):
    """Proxy patches for the band legend, or [] if the band is disabled."""
    if sigma <= 0:
        return []
    handles = [mpatches.Patch(color=_BAND_COLORS["combined"], alpha=0.5,
                              label=_BAND_LABELS["combined"])]
    if visualize_noise:
        handles += [mpatches.Patch(color=_BAND_COLORS[s], alpha=0.5,
                                   label=_BAND_LABELS[s])
                    for s in _OVERLAY_SOURCES]
    return handles


def plot_summary(config, results):
    """Create the 4-panel summary figure.

    Parameters
    ----------
    config : SNRConfig
        The configuration used.
    results : SweepResult
        Output from SNRCalculator.sweep().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0:2])  # SNR vs exposure time
    ax2 = fig.add_subplot(gs[0, 2:4])  # SNR vs mV
    ax3 = fig.add_subplot(gs[1, 2:4])  # Search rate
    ax4 = fig.add_subplot(gs[1, 0:2])  # Parameters

    _plot_snr_vs_exposure(ax1, results, config)
    _plot_snr_vs_mv(ax2, results, config)
    _plot_search_rate(ax3, results, config)
    _plot_params(ax4, results, config)

    return fig


def _plot_snr_vs_exposure(ax, results, config):
    """SNR vs exposure time for various magnitudes."""
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results.mvs)))
    binning = config.observation.binning
    sigma = config.output.sigma
    viz = config.output.visualize_noise

    for i, mv in enumerate(results.mvs):
        y = results.snr_grid[i, :]
        ax.plot(
            results.exposure_times, y,
            color=colors[i], label=f"mv={mv:.1f}",
        )
        _draw_snr_bands(
            ax, results.exposure_times, y,
            {k: v[i, :] for k, v in results.snr_band_unit.items()},
            sigma, viz,
        )

    # Reference SNR lines
    for snr_val in [0.75, 1, 2, 3, 5]:
        ax.axhline(y=snr_val, color='k', linestyle='--', linewidth=0.5)

    # Confidence level annotations
    x_text = results.exposure_times[-1] * 1.01
    for snr_val, ci in [(0.75, 50.0), (1, 68.3), (2, 99.7), (3, 95.5), (5, 99.9)]:
        ax.text(x_text, snr_val, f'{ci}', va='center', fontsize=5)

    ax.set_ylim([0.1, 500.0])
    ax.set_yscale('log')
    ax.set_ylabel(f'Binned ({binning}x{binning}), Co-Added ({config.observation.num_frames}) SNR')
    ax.set_xlabel('Single Frame Exposure Time [s]')
    mag_legend = ax.legend(loc='lower right', ncol=5, prop={'size': 6})
    band_handles = _band_legend_handles(sigma, viz)
    if band_handles:
        ax.add_artist(mag_legend)
        ax.legend(handles=band_handles, loc='upper left', fontsize=6,
                  title=fr'$\pm${sigma:g}$\sigma$ band', title_fontsize=6)
    ax.grid(True)

    # Add co-added collection time axis if coadding
    num_coadds = config.observation.num_apertures * config.observation.num_frames
    if num_coadds > 1 and config.detector.frame_rate:
        ax.set_xlabel('Via Single Frame Exposure Time [s]')
        ax_top = ax.twiny()
        ticks = ax.get_xticks()
        frame_period = 1.0 / config.detector.frame_rate
        collection_times = (
            config.observation.num_frames * np.maximum(ticks, frame_period)
        )
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels([f"{v:.1f}" for v in collection_times])
        ax_top.set_xlabel('Co-Added Collection Time [s]')


def _plot_snr_vs_mv(ax, results, config):
    """SNR vs apparent magnitude at fixed exposure time."""
    binning = config.observation.binning
    t = config.observation.exposure_time
    sigma = config.output.sigma
    viz = config.output.visualize_noise

    ax.plot(results.mvs, results.snr_at_fixed_t, marker='o', linestyle='-')
    _draw_snr_bands(
        ax, results.mvs, results.snr_at_fixed_t,
        dict(results.snr_band_unit_at_fixed_t),
        sigma, viz,
    )

    for snr_val in [0.75, 1, 3, 5]:
        ax.axhline(y=snr_val, color='k', linestyle='--', linewidth=0.5)

    x_text = results.mvs[-1] + 0.01 * (results.mvs[-1] - results.mvs[0])
    for snr_val, ci in [(0.75, 50.0), (1, 68.3), (3, 99.7), (5, 99.9)]:
        ax.text(x_text, snr_val, f'{ci}', va='center', fontsize=5)

    # Saturation line
    if np.any(results.saturated_at_fixed_t):
        sat_idx = int(np.sum(results.saturated_at_fixed_t)) - 1
        if 0 <= sat_idx < len(results.mvs):
            ax.axvline(x=results.mvs[sat_idx], color='r', linestyle='-')
            ax.text(
                results.mvs[sat_idx] * 0.99, 0.2,
                r'$\leftarrow$saturation',
                ha='right', va='center', fontsize=6, color='red',
            )

    ax.set_ylim([0.1, 500.0])
    ax.set_yscale('log')
    ax.set_ylabel(f'Binned ({binning}x{binning}), Co-Added ({config.observation.num_frames}) SNR')
    ax.set_xlabel(f'Apparent Magnitude, m$_{{V}}$ @ t={t}')
    ax.grid(True)

    band_handles = _band_legend_handles(sigma, viz)
    if band_handles:
        ax.legend(handles=band_handles, loc='upper right', fontsize=6,
                  title=fr'$\pm${sigma:g}$\sigma$ band', title_fontsize=6)

    # Secondary axis: photon rate
    ax_top = ax.twiny()
    bot_ticks = ax.get_xticks()
    pps = 10.0 ** ((results.zero_point - bot_ticks) / 2.5)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(bot_ticks)
    ax_top.set_xticklabels([f"{v:.0e}" for v in pps], fontsize=5)
    ax_top.set_xlabel('Photon Rate [photons/s]', fontsize=7)


def _plot_search_rate(ax, results, config):
    """Search rate vs apparent magnitude."""
    snr_threshold = config.observation.snr_threshold
    sigma = config.output.sigma
    viz = config.output.visualize_noise

    ax.plot(results.mvs, results.search_rates, marker='o', linestyle='-')

    # Search-rate envelope: the SNR band propagated through the detection-
    # threshold crossing that sets the rate, precomputed in SNRCalculator.sweep
    # (where the calculator can root-find the perturbed crossing). Combined band
    # by default, plus each noise source when visualize_noise is on.
    if sigma > 0:
        keys = ["combined"] + (list(_OVERLAY_SOURCES) if viz else [])
        for k in keys:
            band = results.search_rate_band.get(k)
            if band is None:
                continue
            lo, hi = band
            alpha = _ALPHA_COMBINED if k == "combined" else _ALPHA_SOURCE
            ax.fill_between(results.mvs, lo, hi, color=_BAND_COLORS[k],
                            alpha=alpha, linewidth=0, zorder=1)

    ax.set_ylabel(fr'Search Rate @ SNR={snr_threshold} [deg$^2$/hr]')
    ax.set_xlabel(r'Apparent Magnitude, m$_{V}$')

    band_handles = _band_legend_handles(sigma, viz)
    if band_handles:
        ax.legend(handles=band_handles, loc='upper right', fontsize=6,
                  title=fr'$\pm${sigma:g}$\sigma$ band', title_fontsize=6)

    # Image circle annotation
    if config.optics.image_circle is not None:
        y_min, y_max = ax.get_ylim()
        x0 = results.mvs[0]
        ax.text(x0, y_min + 0.30 * (y_max - y_min),
                f'Image Circle = {config.optics.image_circle} mm')
        diag = 1e3 * config.detector.pixel_size * np.sqrt(
            config.detector.height ** 2 + config.detector.width ** 2)
        ax.text(x0, y_min + 0.25 * (y_max - y_min),
                f'Detector Diagonal = {diag:.1f} mm')

    # FOV annotations
    fov_h, fov_w = results.fov
    y_min, y_max = ax.get_ylim()
    x0 = results.mvs[0]
    ax.text(x0, y_min + 0.20 * (y_max - y_min), fr'FOV$_x$ = {fov_w:.2f} deg')
    ax.text(x0, y_min + 0.15 * (y_max - y_min), fr'FOV$_y$ = {fov_h:.2f} deg')
    ax.text(x0, y_min + 0.10 * (y_max - y_min),
            fr'FOV = {fov_w * fov_h:.2f} deg$^2$')

    # Secondary axis: satellite diameter
    ax_top = ax.twiny()
    bot_ticks = ax.get_xticks()
    paf_val = float(phase_angle_factor([config.target.phase_angles[0]])[0])
    diameters = 2.0 * config.target.range * np.sqrt(
        10.0 ** ((SUN_MV - bot_ticks) / 2.5)
        / (config.target.albedo * paf_val * np.pi)
    )
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(bot_ticks)
    ax_top.set_xticklabels([f"{d:.1f}" for d in diameters], fontsize=5)
    ax_top.set_xlabel(
        f'Satellite Diameter [m] at {config.target.range / 1e3:.0f} km, '
        rf'$\phi$={config.target.phase_angles[0]:.0f}$^\circ$, '
        f'a={config.target.albedo}',
        fontsize=7,
    )


def _plot_params(ax, results, config):
    """Display parameters as text in the lower-left panel."""
    ax.set_xticks([])
    ax.set_yticks([])

    filter_name = config.filter.name or config.filter.file or 'Open'

    params = [
        f"Telescope: {config.optics.name}",
        f"Detector: {config.detector.name}",
        f"IFOV: {results.ifov:.2f} arcsec, "
        f"binned: {results.ifov * config.observation.binning:.2f} arcsec",
        f"Combined PSF (FWHM): {results.fwhm_arcsec:.1f} arcsec",
        f"Filter: {filter_name}",
        f"# Apertures: {config.observation.num_apertures}",
        f"# Frames: {config.observation.num_frames}",
    ]

    if config.atmosphere.enabled:
        params.extend([
            rf"Sky Background: {config.atmosphere.sky_bkg_mv:.1f} mag/arcsec$^2$",
            f"Atmospheric Transmission: {config.atmosphere.transmission:.2f}",
            rf"r$_0$: {100 * config.atmosphere.r0:.1f} cm",
        ])
    else:
        params.append("Space-based (no atmosphere)")

    if config.target.thermal.enabled:
        params.append(
            f"Target T: {config.target.thermal.temperature:.0f} K, "
            f"\u03b5: {config.target.thermal.emissivity}"
        )

    y_start = 0.92
    spacing = 0.07
    for i, line in enumerate(params):
        ax.text(
            0.05, y_start - i * spacing, line,
            transform=ax.transAxes, va='top', ha='left',
            fontfamily='monospace', fontsize=9,
        )
