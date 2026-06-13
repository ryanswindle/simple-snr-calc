"""Tests for the senpai search-rate overlay (simple_snr_calc.overlay).

These exercise the condition mapping and the model/plot pipeline without
requiring senpai or real night data -- plot_data is plain JSON, so a small
synthetic dict stands in for senpai's output.
"""

import csv
import math

import numpy as np
import pytest

from simple_snr_calc.atmosphere import seeing_sigma_rad
from simple_snr_calc.optics import compute_ifov
from simple_snr_calc.overlay import (
    NightConditions,
    apply_conditions,
    load_nights_summary,
    model_search_rate_vs_mag,
    plot_search_rate_overlay,
    r0_from_fwhm,
)

# A senpai nights_summary.csv row (note the unicode headers senpai emits).
_SUMMARY_ROW = {
    "night": "DAO01_20260602",
    "moon%": "0.92",
    "moonSep°": "61.4",
    "k": "0.097",
    "T_zen": "0.91",
    "clear%": "0.27",
    "FWHM_px": "11.7",
    "FWHM_sd": "5.8",
    "sky_ADU": "3311",
    "sky_μ": "18.57",
    "lim50": "16.55",
    "nFrm": "1393",
}


def _synthetic_plot_data(night_id="DAO01_20260602"):
    """Minimal senpai-shaped plot_data with a 'search_rate' block."""
    mags = list(np.linspace(8.0, 19.0, 80))
    rates = [max(0.0, 1800.0 * (1 - (m - 8) / 11)) for m in mags]
    return {
        "version": 1,
        "meta": {"night_id": night_id, "moon_illumination": 0.92},
        "plots": {
            "search_rate": {
                "mags": mags,
                "rates": rates,
                "binned": {"x": [10.0, 14.0], "y": [1600.0, 800.0],
                           "err_lo": [50.0, 40.0], "err_hi": [60.0, 45.0]},
                "median_lim50": 16.55,
                "n_stars": len(mags),
                "target_snr": 3.0,
                "noise_floor_snr": 0.9,
                "overhead_s": 8.2,
                "overhead_model": None,
            }
        },
    }


class TestNightConditions:

    def test_from_summary_row(self):
        c = NightConditions.from_summary_row(_SUMMARY_ROW)
        assert c.night_id == "DAO01_20260602"
        assert c.zenith_transmission == pytest.approx(0.91)
        assert c.sky_mag_arcsec2 == pytest.approx(18.57)
        assert c.fwhm_px == pytest.approx(11.7)
        assert c.limiting_mag_50 == pytest.approx(16.55)
        assert c.moon_illumination == pytest.approx(0.92)

    def test_blank_cells_become_none(self):
        row = dict(_SUMMARY_ROW, T_zen="", **{"sky_μ": "—"})
        c = NightConditions.from_summary_row(row)
        assert c.zenith_transmission is None
        assert c.sky_mag_arcsec2 is None

    def test_load_nights_summary(self, tmp_path):
        p = tmp_path / "nights_summary.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_SUMMARY_ROW))
            w.writeheader()
            w.writerow(_SUMMARY_ROW)
        out = load_nights_summary(p)
        assert "DAO01_20260602" in out
        assert out["DAO01_20260602"].zenith_transmission == pytest.approx(0.91)


class TestR0FromFwhm:

    @pytest.mark.parametrize("fwhm_arcsec", [0.8, 2.5, 5.0, 11.7])
    def test_roundtrip_through_seeing_model(self, fwhm_arcsec):
        """r0_from_fwhm must invert simple-snr-calc's own seeing relation."""
        wl = 0.65e-6
        r0 = r0_from_fwhm(fwhm_arcsec, wl)
        sigma_rad = seeing_sigma_rad(r0, wl)
        fwhm_rad = sigma_rad * 2.0 * math.sqrt(2.0 * math.log(2.0))
        recovered = math.degrees(fwhm_rad) * 3600.0
        assert recovered == pytest.approx(fwhm_arcsec, rel=1e-9)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            r0_from_fwhm(0.0, 0.65e-6)


class TestApplyConditions:

    def test_overrides_applied(self, ground_config):
        c = NightConditions.from_summary_row(_SUMMARY_ROW)
        cfg = apply_conditions(ground_config, c, overhead_s=8.2,
                               target_snr=3.0, mv_range=(8, 19))
        assert cfg.atmosphere.transmission == pytest.approx(0.91)
        assert cfg.atmosphere.sky_bkg_mv == pytest.approx(18.57)
        assert cfg.optics.step_settle_time == pytest.approx(8.2)
        assert cfg.observation.snr_threshold == pytest.approx(3.0)
        assert cfg.observation.mv_range == [8.0, 19.0]

    def test_fwhm_maps_to_measured_total(self, ground_config):
        """r0 + zeroed jitter must reproduce the measured FWHM exactly."""
        c = NightConditions.from_summary_row(_SUMMARY_ROW)
        cfg = apply_conditions(ground_config, c)
        assert cfg.optics.jitter == 0.0
        ifov = compute_ifov(cfg.detector.pixel_size, cfg.optics.focal_length)
        expected_arcsec = c.fwhm_px * ifov
        sigma = seeing_sigma_rad(cfg.atmosphere.r0, cfg.atmosphere.r0_wavelength)
        fwhm = math.degrees(sigma * 2.355) * 3600.0
        assert fwhm == pytest.approx(expected_arcsec, rel=1e-3)

    def test_does_not_mutate_input(self, ground_config):
        before = ground_config.atmosphere.transmission
        c = NightConditions.from_summary_row(_SUMMARY_ROW)
        apply_conditions(ground_config, c)
        assert ground_config.atmosphere.transmission == before
        assert ground_config.optics.jitter != 0.0  # original jitter untouched

    def test_missing_fields_keep_defaults(self, ground_config):
        c = NightConditions(night_id="x")  # all conditions None
        cfg = apply_conditions(ground_config, c)
        assert cfg.atmosphere.transmission == ground_config.atmosphere.transmission
        assert cfg.atmosphere.sky_bkg_mv == ground_config.atmosphere.sky_bkg_mv
        assert cfg.optics.jitter == ground_config.optics.jitter


class TestModelSearchRate:

    def test_nonneg_and_falls_off(self, ground_config):
        ground_config.observation.snr_threshold = 3.0
        mvs, rates, info = model_search_rate_vs_mag(ground_config, mv_step=0.5)
        assert (rates >= 0).all()
        # Bright objects survey at least as fast as faint ones.
        assert rates[0] >= rates[-1]
        for key in ("fov_sq_deg", "fwhm_arcsec", "zero_point", "snr_threshold"):
            assert key in info
        assert info["snr_threshold"] == pytest.approx(3.0)


class TestPlotOverlay:

    def test_writes_png(self, tmp_path, ground_config):
        summary = {"DAO01_20260602":
                   NightConditions.from_summary_row(_SUMMARY_ROW)}
        out = tmp_path / "overlay.png"
        path = plot_search_rate_overlay(
            _synthetic_plot_data(), summary, ground_config, out, mv_step=0.5
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_unknown_night_raises(self, ground_config):
        with pytest.raises(KeyError):
            plot_search_rate_overlay(
                _synthetic_plot_data("MISSING_NIGHT"), {}, ground_config,
                "/tmp/_unused.png",
            )
