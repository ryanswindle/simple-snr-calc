"""Tests for the multiplicative noise terms: scintillation + systematic floor.

Scintillation and the systematic floor are *signal-proportional* noise: they add
``sigma * S`` electrons to a source of ``S`` electrons. Unlike shot/sky/read
noise they do not average down with brightness, so they impose a magnitude-
independent SNR ceiling of ``1/sigma``. Scintillation (Young 1967 / Osborn 2015)
scales as ``sigma ∝ sqrt(D^(-4/3) X^3 exp(-2h/H) / t)``, so its ceiling fans out
as ``sqrt(t)`` between exposures -- the bright-end flattening seen on-sky and
absent from a pure shot/sky/read budget. The systematic floor is the same kind of
term but exposure-independent.
"""

import numpy as np
import pytest

from simple_snr_calc.atmosphere import (
    SCINTILLATION_SCALE_HEIGHT,
    scintillation_fraction,
)
from simple_snr_calc.noise import (
    MULTIPLICATIVE_SOURCES,
    NOISE_SOURCES,
    compute_noise,
    snr_band_unit,
)
from simple_snr_calc.snr import SNRCalculator


def _budget(**kw):
    params = dict(
        peak_signal=100.0, sky_bkg_rate=5.0, dark_current=0.1, read_noise=2.0,
        full_well=80000.0, bit_depth=16, exposure_time=1.0, binning=1,
        is_cmos=True,
    )
    params.update(kw)
    return compute_noise(**params)


# ── scintillation_fraction: the Young/Osborn physics ─────────────────────────

class TestScintillationFraction:

    def test_disabled_returns_zero(self):
        assert scintillation_fraction(1.0, 1.0, coeff=0.0) == 0.0
        assert scintillation_fraction(1.0, 0.0, coeff=1.5) == 0.0
        assert scintillation_fraction(0.0, 1.0, coeff=1.5) == 0.0

    def test_scales_as_inverse_sqrt_exposure(self):
        """sigma ∝ t^(-1/2): quadrupling t halves the scintillation noise."""
        s1 = scintillation_fraction(1.0, 1.0, coeff=1.5)
        s4 = scintillation_fraction(1.0, 4.0, coeff=1.5)
        assert s1 / s4 == pytest.approx(2.0)

    def test_scales_with_aperture_minus_four_thirds(self):
        """sigma^2 ∝ D^(-4/3): a bigger aperture scintillates less."""
        small = scintillation_fraction(0.5, 1.0, coeff=1.5)
        big = scintillation_fraction(2.0, 1.0, coeff=1.5)
        ratio = (small / big) ** 2
        assert ratio == pytest.approx((2.0 / 0.5) ** (4.0 / 3.0))

    def test_scales_with_airmass_cubed(self):
        """sigma^2 ∝ X^3."""
        z = scintillation_fraction(1.0, 1.0, airmass=1.0, coeff=1.5)
        x2 = scintillation_fraction(1.0, 1.0, airmass=2.0, coeff=1.5)
        assert (x2 / z) ** 2 == pytest.approx(2.0 ** 3)

    def test_altitude_attenuates(self):
        """Higher observatory -> exp(-2h/H) -> less scintillation."""
        sea = scintillation_fraction(1.0, 1.0, observatory_altitude=0.0, coeff=1.5)
        high = scintillation_fraction(1.0, 1.0,
                                      observatory_altitude=4000.0, coeff=1.5)
        assert high < sea
        expected = np.exp(-2.0 * 4000.0 / SCINTILLATION_SCALE_HEIGHT)
        assert (high / sea) ** 2 == pytest.approx(expected)

    def test_realistic_amplitude(self):
        """A 1 m aperture at 1 s, zenith, sea level lands at a few mmag."""
        sigma = scintillation_fraction(1.0, 1.0, coeff=1.5)
        assert 1e-3 < sigma < 1e-2  # ~3 mmag


# ── compute_noise: the terms enter the budget correctly ──────────────────────

class TestNoiseBudget:

    def test_off_by_default_is_backward_compatible(self):
        """No scint/sys args -> identical total to the additive-only budget."""
        nb = _budget()
        assert nb.scintillation == 0.0
        assert nb.systematic == 0.0
        additive = np.sqrt(nb.target_shot**2 + nb.sky_background**2
                           + nb.dark_current**2 + nb.read**2
                           + nb.quantization**2)
        assert nb.total == pytest.approx(additive)

    def test_terms_are_fraction_of_signal(self):
        nb = _budget(peak_signal=1000.0, scint_fraction=0.02,
                     systematic_fraction=0.01)
        assert nb.scintillation == pytest.approx(0.02 * 1000.0)
        assert nb.systematic == pytest.approx(0.01 * 1000.0)

    def test_added_in_quadrature_to_total(self):
        base = _budget(peak_signal=1000.0)
        nb = _budget(peak_signal=1000.0, scint_fraction=0.02,
                     systematic_fraction=0.01)
        expected = np.sqrt(base.total**2 + nb.scintillation**2 + nb.systematic**2)
        assert nb.total == pytest.approx(expected)

    def test_variance_fractions_sum_to_one_with_multiplicative(self):
        nb = _budget(peak_signal=1e5, scint_fraction=0.02,
                     systematic_fraction=0.01)
        vf = nb.variance_fractions()
        assert set(vf) == set((*NOISE_SOURCES, *MULTIPLICATIVE_SOURCES))
        assert sum(vf.values()) == pytest.approx(1.0)
        # At high signal the multiplicative terms dominate the power budget.
        assert vf["scintillation"] + vf["systematic"] > 0.9


# ── The SNR ceiling and its sqrt(t) fan-out ──────────────────────────────────

class TestSNRCeiling:

    def test_bright_snr_approaches_one_over_sigma(self):
        """A bright source is multiplicative-noise limited: SNR -> 1/sigma."""
        f = 0.01
        nb = compute_noise(
            peak_signal=1e10, sky_bkg_rate=5.0, dark_current=0.1,
            read_noise=2.0, full_well=0.0, bit_depth=16, exposure_time=1.0,
            binning=1, is_cmos=True, scint_fraction=f,
        )
        snr = 1e10 / nb.total
        assert snr == pytest.approx(1.0 / f, rel=1e-3)

    def test_combined_sigma_adds_in_quadrature(self):
        """Scintillation + systematic -> ceiling 1/sqrt(fs^2 + fsys^2)."""
        fs, fsys = 0.01, 0.005
        nb = compute_noise(
            peak_signal=1e12, sky_bkg_rate=0.0, dark_current=0.0,
            read_noise=0.0, full_well=0.0, bit_depth=16, exposure_time=1.0,
            binning=1, is_cmos=True, scint_fraction=fs, systematic_fraction=fsys,
        )
        snr = 1e12 / nb.total
        assert snr == pytest.approx(1.0 / np.hypot(fs, fsys), rel=1e-6)

    def test_ceiling_fans_out_as_sqrt_t(self, ground_config):
        """Bright-end SNR ratio between exposures tracks sqrt(t2/t1)."""
        cfg = ground_config.model_copy(deep=True)
        cfg.atmosphere.scintillation_coeff = 1.5
        calc = SNRCalculator(cfg)
        mv = -5.0  # extremely bright -> scintillation limited (SNR -> 1/sigma)
        s1 = calc.compute_snr(mv, 1.0).snr
        s9 = calc.compute_snr(mv, 9.0).snr
        assert s9 / s1 == pytest.approx(3.0, rel=1e-2)  # sqrt(9)

    def test_scintillation_lowers_bright_snr(self, ground_config):
        """Turning scintillation on can only reduce the bright-end SNR."""
        off = ground_config.model_copy(deep=True)
        off.atmosphere.scintillation_coeff = 0.0
        on = ground_config.model_copy(deep=True)
        on.atmosphere.scintillation_coeff = 1.5
        mv = 8.0
        snr_off = SNRCalculator(off).compute_snr(mv, 1.0).snr
        snr_on = SNRCalculator(on).compute_snr(mv, 1.0).snr
        assert snr_on < snr_off


# ── Gating: scintillation is atmospheric, the floor is instrumental ──────────

class TestGating:

    def test_no_scintillation_in_space(self, ground_config):
        """atmosphere.enabled=False -> scintillation term is zero."""
        cfg = ground_config.model_copy(deep=True)
        cfg.atmosphere.enabled = False
        cfg.atmosphere.scintillation_coeff = 1.5
        r = SNRCalculator(cfg).compute_snr(8.0, 1.0)
        assert r.noise.scintillation == 0.0

    def test_systematic_floor_applies_in_space(self, ground_config):
        """The instrumental floor is not gated on the atmosphere."""
        cfg = ground_config.model_copy(deep=True)
        cfg.atmosphere.enabled = False
        cfg.detector.systematic_floor = 0.01
        r = SNRCalculator(cfg).compute_snr(8.0, 1.0)
        assert r.noise.systematic > 0.0


# ── SNR error band: multiplicative noise contributes zero ────────────────────

class TestBandCancellation:

    def test_multiplicative_sources_absent_from_band(self):
        bu = snr_band_unit(_budget(scint_fraction=0.02, systematic_fraction=0.01))
        for s in MULTIPLICATIVE_SOURCES:
            assert s not in bu

    def test_band_collapses_when_scintillation_limited(self):
        """A scintillation-limited SNR is essentially exact: combined -> 0."""
        nb = _budget(peak_signal=1e8, sky_bkg_rate=0.0, dark_current=0.0,
                     read_noise=0.0, scint_fraction=0.05)
        bu = snr_band_unit(nb)
        assert bu["combined"] < 1e-3       # below the usual 1/2 floor
        assert all(np.isfinite(v) for v in bu.values())

    def test_combined_shrinks_when_scintillation_added(self):
        """Adding multiplicative noise inflates total, shrinking every share."""
        base = _budget(peak_signal=1e4)
        scint = _budget(peak_signal=1e4, scint_fraction=0.05)
        assert snr_band_unit(scint)["combined"] < snr_band_unit(base)["combined"]
