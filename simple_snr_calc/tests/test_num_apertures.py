"""Tests for multiple co-boresighted apertures (num_apertures).

Multiple identical co-boresighted telescopes observe the same field.
Signal is independent per aperture, so SNR scales as sqrt(num_apertures).
Search rate should NOT increase with num_apertures (same FOV, just deeper).
"""

import numpy as np
import pytest

from simple_snr_calc.config import SNRConfig, ObservationConfig
from simple_snr_calc.snr import SNRCalculator


class TestNumApertures:

    def test_snr_scales_as_sqrt_n(self, ground_config):
        """SNR should scale as sqrt(num_apertures) for co-boresighted telescopes."""
        ground_config.observation.num_apertures = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        for n in [2, 4, 9]:
            ground_config.observation.num_apertures = n
            calc_n = SNRCalculator(ground_config)
            result_n = calc_n.compute_snr(mv=17.0, exposure_time=1.0)

            expected_ratio = np.sqrt(n)
            actual_ratio = result_n.snr / result_1.snr

            np.testing.assert_allclose(
                actual_ratio, expected_ratio, rtol=1e-10,
                err_msg=f"SNR ratio for {n} apertures should be sqrt({n})",
            )

    def test_peak_signal_unchanged(self, ground_config):
        """Peak signal per frame should not change with num_apertures."""
        ground_config.observation.num_apertures = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_apertures = 4
        calc_4 = SNRCalculator(ground_config)
        result_4 = calc_4.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_4.peak_signal, result_1.peak_signal, rtol=1e-10,
            err_msg="Peak signal should not change with num_apertures",
        )

    def test_search_rate_increases_with_apertures(self, ground_config):
        """More apertures reach the SNR threshold faster, increasing search rate."""
        ground_config.observation.num_apertures = 1
        calc_1 = SNRCalculator(ground_config)
        results_1 = calc_1.sweep()

        ground_config.observation.num_apertures = 4
        calc_4 = SNRCalculator(ground_config)
        results_4 = calc_4.sweep()

        # For magnitudes where both have nonzero search rate,
        # more apertures should give equal or better search rate
        mask = (results_1.search_rates > 0) & (results_4.search_rates > 0)
        assert np.all(results_4.search_rates[mask] >= results_1.search_rates[mask]), \
            "More apertures should give equal or better search rate"

    def test_noise_budget_unchanged(self, ground_config):
        """Individual noise components should not change with num_apertures."""
        ground_config.observation.num_apertures = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_apertures = 4
        calc_4 = SNRCalculator(ground_config)
        result_4 = calc_4.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_4.noise.total, result_1.noise.total, rtol=1e-10,
            err_msg="Per-frame noise should not change with num_apertures",
        )
