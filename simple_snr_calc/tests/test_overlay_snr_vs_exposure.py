"""Tests for the senpai SNR-vs-exposure (by-magnitude) overlay."""

import numpy as np
import pytest

from simple_snr_calc.overlay import (
    NightConditions,
    model_snr_vs_exposure,
    plot_snr_vs_exposure_overlay,
)

_SUMMARY_ROW = {
    "night": "DAO01_20260602", "moon%": "0.92", "moonSep°": "61.4",
    "k": "0.097", "T_zen": "0.91", "FWHM_px": "11.7", "sky_μ": "18.57",
    "lim50": "16.55",
}


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

    def test_writes_png(self, tmp_path, ground_config):
        summary = {"DAO01_20260602":
                   NightConditions.from_summary_row(_SUMMARY_ROW)}
        out = tmp_path / "snr_vs_exposure.png"
        path = plot_snr_vs_exposure_overlay(
            _synthetic_plot_data(), summary, ground_config, out)
        assert path.exists() and path.stat().st_size > 0

    def test_missing_block_raises(self, ground_config):
        pd = _synthetic_plot_data()
        del pd["plots"]["snr_vs_exposure"]
        with pytest.raises(ValueError):
            plot_snr_vs_exposure_overlay(pd, {}, ground_config, "/tmp/_x.png")

    def test_no_pooled_raises(self, ground_config):
        summary = {"DAO01_20260602":
                   NightConditions.from_summary_row(_SUMMARY_ROW)}
        with pytest.raises(ValueError):
            plot_snr_vs_exposure_overlay(
                _synthetic_plot_data(pooled=False), summary,
                ground_config, "/tmp/_x.png")

    def test_unknown_night_raises(self, ground_config):
        with pytest.raises(KeyError):
            plot_snr_vs_exposure_overlay(
                _synthetic_plot_data("MISSING"), {}, ground_config, "/tmp/_x.png")
