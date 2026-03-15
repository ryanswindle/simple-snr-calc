"""Tests for detector binning (CCD vs CMOS).

For NxN binning (N = binning factor, N^2 pixels summed):
- Signal: N^2 x single-pixel signal
- CCD read noise: unchanged (single readout after on-chip charge transfer)
- CMOS read noise: N x single-pixel read noise (N^2 independent reads, RSS)
- Sky background: scales as N^2 (solid angle scales as N^2)
- Dark current: scales as N^2 (N^2 pixels contribute)
"""

import numpy as np
import pytest

from simple_snr_calc.noise import compute_noise, NoiseBudget


class TestBinningNoise:

    def test_cmos_read_noise_scales_as_binning(self):
        """CMOS read noise RMS should scale as N (binning factor)."""
        base = compute_noise(
            peak_signal=1000, sky_bkg_rate=10, dark_current=0.1,
            read_noise=3.0, full_well=80000, bit_depth=16,
            exposure_time=1.0, binning=1, is_cmos=True,
        )

        for N in [2, 3, 4]:
            binned = compute_noise(
                peak_signal=1000, sky_bkg_rate=10, dark_current=0.1,
                read_noise=3.0, full_well=80000, bit_depth=16,
                exposure_time=1.0, binning=N, is_cmos=True,
            )

            np.testing.assert_allclose(
                binned.read, base.read * N, rtol=1e-10,
                err_msg=f"CMOS read noise should scale as {N} for {N}x{N} binning",
            )

    def test_ccd_read_noise_unchanged(self):
        """CCD read noise should not change with binning."""
        base = compute_noise(
            peak_signal=1000, sky_bkg_rate=10, dark_current=0.1,
            read_noise=3.0, full_well=80000, bit_depth=16,
            exposure_time=1.0, binning=1, is_cmos=False,
        )

        for N in [2, 3, 4]:
            binned = compute_noise(
                peak_signal=1000, sky_bkg_rate=10, dark_current=0.1,
                read_noise=3.0, full_well=80000, bit_depth=16,
                exposure_time=1.0, binning=N, is_cmos=False,
            )

            np.testing.assert_allclose(
                binned.read, base.read, rtol=1e-10,
                err_msg=f"CCD read noise should not change with {N}x{N} binning",
            )

    def test_ccd_better_snr_than_cmos_when_read_dominated(self):
        """CCD should have better SNR than CMOS for read-noise-dominated regime."""
        # Low signal, high read noise -> read-noise dominated
        ccd = compute_noise(
            peak_signal=10, sky_bkg_rate=0.1, dark_current=0.01,
            read_noise=10.0, full_well=80000, bit_depth=16,
            exposure_time=1.0, binning=3, is_cmos=False,
        )
        cmos = compute_noise(
            peak_signal=10, sky_bkg_rate=0.1, dark_current=0.01,
            read_noise=10.0, full_well=80000, bit_depth=16,
            exposure_time=1.0, binning=3, is_cmos=True,
        )

        assert ccd.total < cmos.total, \
            "CCD should have less total noise than CMOS when read-noise dominated"

    def test_ccd_cmos_identical_at_binning_1(self):
        """CCD and CMOS should have identical noise at binning=1."""
        ccd = compute_noise(
            peak_signal=500, sky_bkg_rate=10, dark_current=0.1,
            read_noise=3.0, full_well=80000, bit_depth=16,
            exposure_time=1.0, binning=1, is_cmos=False,
        )
        cmos = compute_noise(
            peak_signal=500, sky_bkg_rate=10, dark_current=0.1,
            read_noise=3.0, full_well=80000, bit_depth=16,
            exposure_time=1.0, binning=1, is_cmos=True,
        )

        np.testing.assert_allclose(
            ccd.total, cmos.total, rtol=1e-10,
            err_msg="CCD and CMOS noise should be identical at binning=1",
        )


class TestBinningSkyBackground:

    def test_sky_background_scales_with_binning_squared(self, ground_config):
        """Sky background rate should scale as binning^2 (solid angle)."""
        from simple_snr_calc.snr import SNRCalculator

        ground_config.observation.binning = 1
        calc_1 = SNRCalculator(ground_config)
        sky_1 = calc_1.sky_bkg_rate

        for N in [2, 3]:
            ground_config.observation.binning = N
            calc_n = SNRCalculator(ground_config)
            sky_n = calc_n.sky_bkg_rate

            np.testing.assert_allclose(
                sky_n / sky_1, N ** 2, rtol=1e-6,
                err_msg=f"Sky background should scale as {N}^2 for {N}x{N} binning",
            )
