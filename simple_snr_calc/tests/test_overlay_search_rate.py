"""Tests for the senpai search-rate overlay (simple_snr_calc.overlay).

These exercise the condition mapping and the model/plot pipeline without
requiring senpai or real night data -- plot_data is plain JSON, so a small
synthetic dict stands in for senpai's output.
"""

import json
import math

import numpy as np
import pytest

from simple_snr_calc.atmosphere import seeing_sigma_rad
from simple_snr_calc.optics import compute_ifov
from simple_snr_calc.overlay import (
    NightConditions,
    apply_conditions,
    load_night_conditions,
    model_search_rate_vs_mag,
    plot_search_rate_overlay,
    r0_from_fwhm,
)


def _cond(**overrides):
    """A night's measured conditions, as load_night_conditions would yield."""
    base = dict(night_id="DAO01_20260602", extinction_k=0.097,
                zenith_transmission=0.91, sky_mag_arcsec2=18.57, fwhm_px=11.7,
                limiting_mag_50=16.55, moon_illumination=0.92, moon_sep_deg=61.4)
    base.update(overrides)
    return NightConditions(**base)


def _night_calibration(night_id="DAO01_20260602", **cond_overrides):
    """A minimal senpai ``night_calibration.json`` dict (the 'conditions' block)."""
    conditions = {
        "moon_illumination": 0.92,
        "moon_sep_median_deg": 61.4,
        "extinction_k": 0.097,
        "zenith_transmission": 0.91,
        "fwhm_px_median": 11.7,
        "sky_mag_arcsec2_median": 18.57,
        "limiting_mag_50_median": 16.55,
    }
    conditions.update(cond_overrides)
    return {"night_id": night_id,
            "moon_illumination": conditions["moon_illumination"],
            "conditions": conditions}


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

    def test_load_night_conditions(self, tmp_path):
        p = tmp_path / "night_calibration.json"
        p.write_text(json.dumps(_night_calibration()))
        c = load_night_conditions(p)
        assert c.night_id == "DAO01_20260602"
        assert c.zenith_transmission == pytest.approx(0.91)
        assert c.sky_mag_arcsec2 == pytest.approx(18.57)
        assert c.fwhm_px == pytest.approx(11.7)
        assert c.limiting_mag_50 == pytest.approx(16.55)
        assert c.moon_illumination == pytest.approx(0.92)
        assert c.moon_sep_deg == pytest.approx(61.4)

    def test_missing_measurements_become_none(self, tmp_path):
        cal = _night_calibration()
        del cal["conditions"]["zenith_transmission"]
        cal["conditions"]["sky_mag_arcsec2_median"] = None
        p = tmp_path / "night_calibration.json"
        p.write_text(json.dumps(cal))
        c = load_night_conditions(p)
        assert c.zenith_transmission is None
        assert c.sky_mag_arcsec2 is None


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
        c = _cond()
        cfg = apply_conditions(ground_config, c, overhead_s=8.2,
                               target_snr=3.0, mv_range=(8, 19))
        assert cfg.atmosphere.transmission == pytest.approx(0.91)
        assert cfg.atmosphere.sky_bkg_mv == pytest.approx(18.57)
        assert cfg.optics.step_settle_time == pytest.approx(8.2)
        assert cfg.observation.snr_threshold == pytest.approx(3.0)
        assert cfg.observation.mv_range == [8.0, 19.0]

    def test_fwhm_maps_to_measured_total(self, ground_config):
        """r0 + zeroed jitter must reproduce the measured FWHM exactly."""
        c = _cond()
        cfg = apply_conditions(ground_config, c)
        assert cfg.optics.jitter == 0.0
        ifov = compute_ifov(cfg.detector.pixel_size, cfg.optics.focal_length)
        expected_arcsec = c.fwhm_px * ifov
        sigma = seeing_sigma_rad(cfg.atmosphere.r0, cfg.atmosphere.r0_wavelength)
        fwhm = math.degrees(sigma * 2.355) * 3600.0
        assert fwhm == pytest.approx(expected_arcsec, rel=1e-3)

    def test_does_not_mutate_input(self, ground_config):
        before = ground_config.atmosphere.transmission
        c = _cond()
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

    def test_writes_titled_and_clean(self, tmp_path, ground_config):
        out = tmp_path / "overlay.png"
        paths = plot_search_rate_overlay(
            _synthetic_plot_data(), _cond(), ground_config, out, mv_step=0.5
        )
        assert [p.name for p in paths] == ["overlay.png", "overlay_clean.png"]
        for p in paths:
            assert p.exists() and p.stat().st_size > 0

    def test_accepts_calibration_json_path(self, tmp_path, ground_config):
        """A path to night_calibration.json is loaded directly -- no CSV needed."""
        p = tmp_path / "night_calibration.json"
        p.write_text(json.dumps(_night_calibration()))
        out = tmp_path / "overlay_from_path.png"
        paths = plot_search_rate_overlay(
            _synthetic_plot_data(), p, ground_config, out, mv_step=0.5)
        assert all(q.exists() and q.stat().st_size > 0 for q in paths)


class TestSaveOverlayVariants:

    def test_writes_clean_twin_without_title(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from simple_snr_calc.overlay.search_rate import save_overlay_variants

        fig, ax = plt.subplots()
        ax.set_title("a title")
        out = tmp_path / "fig.png"
        paths = save_overlay_variants(fig, ax, out)
        plt.close(fig)
        assert [p.name for p in paths] == ["fig.png", "fig_clean.png"]
        assert all(p.exists() for p in paths)
        assert ax.get_title() == ""  # clean twin had its title stripped
