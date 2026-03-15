"""Tests for multiple non-overlapping fields (num_fields).

Multiple telescopes with non-overlapping FOVs. Each field observes
independently, so SNR per field is unchanged. Search rate scales
linearly with num_fields (more sky covered per unit time).
"""

import numpy as np
import pytest

from simple_snr_calc.config import SNRConfig, ObservationConfig
from simple_snr_calc.snr import SNRCalculator


class TestNumFields:

    def test_snr_unchanged(self, ground_config):
        """SNR should not change with num_fields (independent FOVs)."""
        ground_config.observation.num_fields = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_fields = 4
        calc_4 = SNRCalculator(ground_config)
        result_4 = calc_4.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_4.snr, result_1.snr, rtol=1e-10,
            err_msg="SNR should not change with num_fields",
        )

    def test_search_rate_scales_linearly(self, ground_config):
        """Search rate should scale linearly with num_fields."""
        ground_config.observation.num_fields = 1
        calc_1 = SNRCalculator(ground_config)
        results_1 = calc_1.sweep()

        for n in [2, 3, 5]:
            ground_config.observation.num_fields = n
            calc_n = SNRCalculator(ground_config)
            results_n = calc_n.sweep()

            # Where both have nonzero search rates, ratio should be n
            mask = results_1.search_rates > 0
            if np.any(mask):
                ratio = results_n.search_rates[mask] / results_1.search_rates[mask]
                np.testing.assert_allclose(
                    ratio, n, rtol=1e-10,
                    err_msg=f"Search rate should scale by {n}x with {n} fields",
                )

    def test_peak_signal_unchanged(self, ground_config):
        """Signal should not change with num_fields."""
        ground_config.observation.num_fields = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_fields = 4
        calc_4 = SNRCalculator(ground_config)
        result_4 = calc_4.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_4.peak_signal, result_1.peak_signal, rtol=1e-10,
        )

    def test_noise_unchanged(self, ground_config):
        """Noise should not change with num_fields."""
        ground_config.observation.num_fields = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_fields = 4
        calc_4 = SNRCalculator(ground_config)
        result_4 = calc_4.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_4.noise.total, result_1.noise.total, rtol=1e-10,
        )
