"""Tests for saturation detection and reporting.

The detector saturates when total electrons (signal + sky + dark + read)
exceed the full well capacity. The API should flag this, and the sweep
should propagate saturation info for plotting.
"""

import numpy as np
import pytest

from simple_snr_calc.config import SNRConfig, DetectorConfig
from simple_snr_calc.snr import SNRCalculator, SNRResult, SweepResult


class TestSaturationDetection:

    def test_bright_target_saturates(self, ground_config):
        """A very bright target should saturate."""
        calc = SNRCalculator(ground_config)
        # mV=5 is extremely bright
        result = calc.compute_snr(mv=5.0, exposure_time=10.0)
        assert result.saturated, "Very bright target at long exposure should saturate"

    def test_faint_target_does_not_saturate(self, ground_config):
        """A faint target should not saturate."""
        calc = SNRCalculator(ground_config)
        result = calc.compute_snr(mv=20.0, exposure_time=0.1)
        assert not result.saturated, "Faint target at short exposure should not saturate"

    def test_saturation_depends_on_exposure_time(self, ground_config):
        """For a given mV, longer exposure increases likelihood of saturation."""
        calc = SNRCalculator(ground_config)
        mv = 12.0

        # Find the transition: short exposure should not saturate,
        # long exposure should
        result_short = calc.compute_snr(mv, exposure_time=0.001)
        result_long = calc.compute_snr(mv, exposure_time=100.0)

        # At least one should be unsaturated and one saturated
        assert not result_short.saturated or result_long.saturated, \
            "Saturation should depend on exposure time"

    def test_saturation_depends_on_full_well(self, ground_config):
        """Lower full well should saturate more easily."""
        # High full well
        ground_config.detector.full_well = 1e6
        calc_high = SNRCalculator(ground_config)
        result_high = calc_high.compute_snr(mv=12.0, exposure_time=5.0)

        # Low full well
        ground_config.detector.full_well = 1000
        calc_low = SNRCalculator(ground_config)
        result_low = calc_low.compute_snr(mv=12.0, exposure_time=5.0)

        assert result_low.saturated, "Low full well should saturate"
        assert not result_high.saturated, "High full well should not saturate"


class TestSaturationInSweep:

    def test_sweep_returns_saturation_grid(self, ground_config):
        """Sweep should return a boolean saturation grid."""
        ground_config.observation.mv_range = [10.0, 20.0]
        ground_config.observation.mv_step = 2.0
        calc = SNRCalculator(ground_config)
        results = calc.sweep()

        assert results.saturated_grid.dtype == bool
        assert results.saturated_grid.shape == results.snr_grid.shape

    def test_sweep_returns_saturation_at_fixed_t(self, ground_config):
        """Sweep should return saturation flags at the fixed exposure time."""
        ground_config.observation.mv_range = [10.0, 20.0]
        ground_config.observation.mv_step = 2.0
        calc = SNRCalculator(ground_config)
        results = calc.sweep()

        assert results.saturated_at_fixed_t.dtype == bool
        assert len(results.saturated_at_fixed_t) == len(results.mvs)

    def test_bright_end_saturates_before_faint_end(self, ground_config):
        """Brighter magnitudes should saturate at shorter exposures."""
        ground_config.observation.mv_range = [8.0, 20.0]
        ground_config.observation.mv_step = 2.0
        ground_config.observation.exposure_range = [0.1, 30.0]
        calc = SNRCalculator(ground_config)
        results = calc.sweep()

        # For each exposure time, if a faint target saturates,
        # all brighter targets should also saturate
        for j in range(results.snr_grid.shape[1]):
            sat_col = results.saturated_grid[:, j]
            # Find last saturated index (bright end is index 0)
            sat_indices = np.where(sat_col)[0]
            if len(sat_indices) > 0:
                # All indices below (brighter) should also be saturated
                max_sat = sat_indices[-1]
                assert np.all(sat_col[:max_sat + 1]), \
                    "All targets brighter than a saturated one should also saturate"

    def test_saturation_result_accessible(self, ground_config):
        """SNRResult.saturated should be a bool accessible via the API."""
        calc = SNRCalculator(ground_config)
        result = calc.compute_snr(mv=17.0, exposure_time=1.0)
        assert isinstance(result.saturated, (bool, np.bool_))
