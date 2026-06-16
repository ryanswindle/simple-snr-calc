"""Tests for the senpai SNR-vs-exposure (by-magnitude) overlay."""

import numpy as np
import pytest

from simple_snr_calc.overlay import (
    NightConditions,
    model_snr_vs_exposure,
    plot_snr_vs_exposure_overlay,
)

def _cond(**overrides):
    """A night's measured conditions, as load_night_conditions would yield."""
    base = dict(night_id="DAO01_20260602", extinction_k=0.097,
                zenith_transmission=0.91, sky_mag_arcsec2=18.57, fwhm_px=11.7,
                limiting_mag_50=16.55, moon_illumination=0.92, moon_sep_deg=61.4)
    base.update(overrides)
    return NightConditions(**base)


def _synthetic_plot_data(night_id="DAO01_20260602", pooled=True):
    exps = [1, 2, 3, 5, 10]
    series = []
    if pooled:
        for lo in (12, 14):  # magnitude bin lower edges
            y = [100.0 * 10 ** (-0.4 * (lo - 12)) * np.sqrt(t) for t in exps]
            series.append({"bin": lo, "x": exps, "y": y,
                           "e_lo": [v * 0.1 for v in y],
                           "e_hi": [v * 0.1 for v in y]})
    return {
        "version": 1,
        "meta": {"night_id": night_id, "moon_illumination": 0.92},
        "plots": {
            "snr_vs_exposure": {
                "std_exps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "bins": [12, 14],
                "order": ["coverage"],
                "faceted": {},
                "pooled": series,
            }
        },
    }


class TestModelSnrVsExposure:

    def test_rises_with_exposure(self, ground_config):
        exps, curves, info = model_snr_vs_exposure(
            ground_config, mags=[14.0, 17.0], exp_min=1, exp_max=10, n=20)
        assert set(curves) == {14.0, 17.0}
        for m in (14.0, 17.0):
            assert len(curves[m]) == len(exps)
            # SNR increases with exposure time.
            assert curves[m][-1] > curves[m][0]
        # Brighter magnitude -> higher SNR at the same exposure.
        assert (curves[14.0] >= curves[17.0]).all()
        assert "zero_point" in info


class TestPlotSnrVsExposure:

    def test_writes_titled_and_clean(self, tmp_path, ground_config):
        out = tmp_path / "snr_vs_exposure.png"
        paths = plot_snr_vs_exposure_overlay(
            _synthetic_plot_data(), _cond(), ground_config, out)
        assert [p.name for p in paths] == [
            "snr_vs_exposure.png", "snr_vs_exposure_clean.png"]
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    def test_missing_block_raises(self, ground_config):
        pd = _synthetic_plot_data()
        del pd["plots"]["snr_vs_exposure"]
        with pytest.raises(ValueError):
            plot_snr_vs_exposure_overlay(pd, _cond(), ground_config, "/tmp/_x.png")

    def test_no_pooled_raises(self, ground_config):
        with pytest.raises(ValueError):
            plot_snr_vs_exposure_overlay(
                _synthetic_plot_data(pooled=False), _cond(),
                ground_config, "/tmp/_x.png")
