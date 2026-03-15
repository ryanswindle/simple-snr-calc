"""Tests for temporal coadding (num_frames).

Coadding N frames of temporally uncorrelated data:
- Signal: N x single-frame signal
- Noise: sqrt(N) x single-frame noise (uncorrelated)
- SNR: sqrt(N) x single-frame SNR
This assumes photon-noise-dominated regime where noise scales as sqrt(signal).
The code applies SNR *= sqrt(num_frames), which is exact for all noise sources
since each frame has independent shot, read, dark, sky, and quantization noise.
"""

import numpy as np
import pytest

from simple_snr_calc.snr import SNRCalculator


class TestCoadding:

    def test_snr_scales_as_sqrt_num_frames(self, ground_config):
        """SNR should scale as sqrt(num_frames) for uncorrelated noise."""
        ground_config.observation.num_frames = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        for n in [4, 9, 16, 25]:
            ground_config.observation.num_frames = n
            calc_n = SNRCalculator(ground_config)
            result_n = calc_n.compute_snr(mv=17.0, exposure_time=1.0)

            expected_ratio = np.sqrt(n)
            actual_ratio = result_n.snr / result_1.snr

            np.testing.assert_allclose(
                actual_ratio, expected_ratio, rtol=1e-10,
                err_msg=f"SNR should scale as sqrt({n}) for {n} coadded frames",
            )

    def test_peak_signal_unchanged_by_coadding(self, ground_config):
        """Per-frame Peak signal should not change with num_frames."""
        ground_config.observation.num_frames = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_frames = 10
        calc_10 = SNRCalculator(ground_config)
        result_10 = calc_10.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_10.peak_signal, result_1.peak_signal, rtol=1e-10,
            err_msg="Peak signal should not change with num_frames",
        )

    def test_noise_budget_unchanged_by_coadding(self, ground_config):
        """Per-frame noise budget should not change with num_frames."""
        ground_config.observation.num_frames = 1
        calc_1 = SNRCalculator(ground_config)
        result_1 = calc_1.compute_snr(mv=17.0, exposure_time=1.0)

        ground_config.observation.num_frames = 10
        calc_10 = SNRCalculator(ground_config)
        result_10 = calc_10.compute_snr(mv=17.0, exposure_time=1.0)

        np.testing.assert_allclose(
            result_10.noise.total, result_1.noise.total, rtol=1e-10,
            err_msg="Per-frame noise should not change with num_frames",
        )

    def test_coadding_and_apertures_combine(self, ground_config):
        """Coadding and multiple apertures should combine multiplicatively."""
        ground_config.observation.num_frames = 1
        ground_config.observation.num_apertures = 1
        calc_base = SNRCalculator(ground_config)
        result_base = calc_base.compute_snr(mv=17.0, exposure_time=1.0)

        n_frames = 4
        n_apertures = 3
        ground_config.observation.num_frames = n_frames
        ground_config.observation.num_apertures = n_apertures
        calc_combined = SNRCalculator(ground_config)
        result_combined = calc_combined.compute_snr(mv=17.0, exposure_time=1.0)

        expected_ratio = np.sqrt(n_frames * n_apertures)
        actual_ratio = result_combined.snr / result_base.snr

        np.testing.assert_allclose(
            actual_ratio, expected_ratio, rtol=1e-10,
            err_msg="SNR should scale as sqrt(num_frames * num_apertures)",
        )

    def test_single_frame_no_coadd_boost(self, ground_config):
        """With num_frames=1 and num_apertures=1, no coadding boost."""
        ground_config.observation.num_frames = 1
        ground_config.observation.num_apertures = 1
        calc = SNRCalculator(ground_config)
        result = calc.compute_snr(mv=17.0, exposure_time=1.0)

        # SNR should equal peak_signal / noise.total (no sqrt(N) boost)
        expected_snr = result.peak_signal / result.noise.total
        np.testing.assert_allclose(
            result.snr, expected_snr, rtol=1e-10,
            err_msg="Single frame SNR should be signal/noise with no coadd boost",
        )
