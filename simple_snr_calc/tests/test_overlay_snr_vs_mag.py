"""Tests for the senpai SNR-vs-magnitude overlay."""

import numpy as np
import pytest

from simple_snr_calc.overlay import (
    NightConditions,
    model_snr_vs_mag,
    plot_snr_vs_mag_overlay,
)

def _cond(**overrides):
    """A night's measured conditions, as load_night_conditions would yield."""
    base = dict(night_id="DAO01_20260602", extinction_k=0.097,
                zenith_transmission=0.91, sky_mag_arcsec2=18.57, fwhm_px=11.7,
                limiting_mag_50=16.55, moon_illumination=0.92, moon_sep_deg=61.4)
    base.update(overrides)
    return NightConditions(**base)


def _synthetic_plot_data(night_id="DAO01_20260602", with_lines=True):
    mags = list(np.arange(10.0, 17.0, 0.5))
    lines = []
    if with_lines:
        for exp in (1, 5, 10):
            # SNR falls with mag, rises with exposure (toy values).
            y = [1000.0 * np.sqrt(exp) * 10 ** (-0.4 * (m - 10)) for m in mags]
            lines.append({"exp": exp, "x": mags, "y": y})
    return {
        "version": 1,
        "meta": {"night_id": night_id, "moon_illumination": 0.92},
        "plots": {
            "snr_vs_mag_weathermasked": {
                "std_exps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "lines": lines,
                "lim50": {"med": 16.55, "lo": 15.85, "hi": 17.33},
                "min_meas_snr": 3.0,
                "zp_mode": 26.36, "zp_sig": 0.2,
            }
        },
    }


class TestModelSnrVsMag:

    def test_shape_and_monotonicity(self, ground_config):
        mvs, curves, info = model_snr_vs_mag(
            ground_config, exposures=[1, 10], mv_min=12, mv_max=18, mv_step=0.5)
        assert set(curves) == {1, 10}
        for t in (1, 10):
            assert len(curves[t]) == len(mvs)
            # SNR decreases as magnitude increases (fainter -> lower SNR).
            assert curves[t][0] > curves[t][-1]
        # Longer exposure -> higher SNR at the same magnitude.
        assert (curves[10] >= curves[1]).all()
        assert "zero_point" in info and "fwhm_arcsec" in info


class TestPlotSnrVsMag:

    def test_writes_titled_and_clean(self, tmp_path, ground_config):
        out = tmp_path / "snr_vs_mag.png"
        paths = plot_snr_vs_mag_overlay(
            _synthetic_plot_data(), _cond(), ground_config, out, mv_step=0.5)
        assert [p.name for p in paths] == [
            "snr_vs_mag.png", "snr_vs_mag_clean.png"]
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)

    def test_missing_plot_block_raises(self, ground_config):
        pd = _synthetic_plot_data()
        del pd["plots"]["snr_vs_mag_weathermasked"]
        with pytest.raises(ValueError):
            plot_snr_vs_mag_overlay(pd, _cond(), ground_config, "/tmp/_x.png")

    def test_no_lines_raises(self, ground_config):
        with pytest.raises(ValueError):
            plot_snr_vs_mag_overlay(
                _synthetic_plot_data(with_lines=False), _cond(),
                ground_config, "/tmp/_x.png")
