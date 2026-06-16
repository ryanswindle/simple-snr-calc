"""Tests for the SNR error bar.

The honest 1-sigma bar on the coadded SNR, propagated through
``SNR = sqrt(Nc) * S / sqrt(S + B)`` with the noise estimate tracking the
measured signal, is

    sigma(SNR) = (S/2 + B)/(S + B) = 1 - f_shot/2 ,

with S = target-shot variance, B = sky+dark+read+quant variance, and
f_shot = (target_shot/total)**2. It includes every source, is nonzero
everywhere (1/2 when target-shot limited, 1 when read/sky/dark/quant limited),
and is coadd-independent. The per-source *contributions* add linearly (each
proportional to that source's variance, target shot down-weighted by 1/2
because it is correlated with the signal) and sum to ``"combined"``, the
default plotted band.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from simple_snr_calc.config import OutputConfig
from simple_snr_calc.noise import NOISE_SOURCES, compute_noise, snr_band_unit
from simple_snr_calc.plotting import plot_summary
from simple_snr_calc.snr import SNRCalculator


def _budget(**kw):
    params = dict(
        peak_signal=100.0, sky_bkg_rate=5.0, dark_current=0.1, read_noise=2.0,
        full_well=80000.0, bit_depth=16, exposure_time=1.0, binning=1,
        is_cmos=True,
    )
    params.update(kw)
    return compute_noise(**params)


# ── snr_band_unit: the propagated bar and its decomposition ──────────────────

class TestBandUnit:

    def test_contributions_sum_linearly_to_combined(self):
        bu = snr_band_unit(_budget())
        assert sum(bu[s] for s in NOISE_SOURCES) == pytest.approx(bu["combined"])

    def test_per_source_formula(self):
        nb = _budget()
        bu = snr_band_unit(nb)
        for s in NOISE_SOURCES:
            var_frac = (getattr(nb, s) / nb.total) ** 2
            expected = 0.5 * var_frac if s == "target_shot" else var_frac
            assert bu[s] == pytest.approx(expected)

    def test_combined_formula(self):
        nb = _budget()
        f_shot = (nb.target_shot / nb.total) ** 2
        assert snr_band_unit(nb)["combined"] == pytest.approx(1.0 - 0.5 * f_shot)

    def test_combined_bounded_half_to_one(self):
        # Sweep a range of regimes; the bar must stay within [1/2, 1].
        for ps in [0.0, 1.0, 1e2, 1e4, 1e8]:
            for rn in [0.0, 1.0, 50.0]:
                bu = snr_band_unit(_budget(peak_signal=ps, read_noise=rn))
                assert 0.5 - 1e-9 <= bu["combined"] <= 1.0 + 1e-9

    def test_half_when_shot_limited(self):
        nb = _budget(peak_signal=1e8, sky_bkg_rate=0.0, dark_current=0.0,
                     read_noise=0.0)
        bu = snr_band_unit(nb)
        assert bu["combined"] == pytest.approx(0.5, abs=1e-3)
        assert bu["target_shot"] == pytest.approx(0.5, abs=1e-3)

    def test_one_when_background_limited(self):
        nb = _budget(peak_signal=1.0, sky_bkg_rate=0.0, dark_current=0.0,
                     read_noise=50.0)
        bu = snr_band_unit(nb)
        assert bu["combined"] == pytest.approx(1.0, abs=1e-2)
        assert bu["read"] == pytest.approx(1.0, abs=1e-2)

    def test_combined_grows_with_dark_current(self):
        base = dict(peak_signal=500.0, sky_bkg_rate=1.0, read_noise=1.0,
                    full_well=80000.0, bit_depth=16, exposure_time=10.0,
                    binning=1, is_cmos=True)
        lo = snr_band_unit(compute_noise(dark_current=0.01, **base))
        hi = snr_band_unit(compute_noise(dark_current=100.0, **base))
        assert hi["combined"] > lo["combined"]
        assert hi["dark_current"] > lo["dark_current"]

    def test_combined_grows_with_lower_bit_depth(self):
        base = dict(peak_signal=500.0, sky_bkg_rate=1.0, dark_current=0.01,
                    read_noise=1.0, full_well=80000.0, exposure_time=1.0,
                    binning=1, is_cmos=True)
        hi_bits = snr_band_unit(compute_noise(bit_depth=16, **base))
        lo_bits = snr_band_unit(compute_noise(bit_depth=8, **base))
        assert lo_bits["combined"] > hi_bits["combined"]
        assert lo_bits["quantization"] > hi_bits["quantization"]

    def test_zero_noise_is_safe(self):
        nb = compute_noise(
            peak_signal=0.0, sky_bkg_rate=0.0, dark_current=0.0,
            read_noise=0.0, full_well=0.0, bit_depth=16, exposure_time=0.0,
            binning=1, is_cmos=True,
        )
        bu = snr_band_unit(nb)
        assert all(np.isfinite(v) for v in bu.values())
        assert all(v == 0.0 for v in bu.values())


# ── Regime structure on a real sweep ─────────────────────────────────────────

class TestRegimes:

    def test_grids_present_and_shaped(self, ground_config):
        r = SNRCalculator(ground_config).sweep()
        n_mv, n_t = r.snr_grid.shape
        for s in (*NOISE_SOURCES, "combined"):
            assert r.snr_band_unit[s].shape == (n_mv, n_t)
            assert r.snr_band_unit_at_fixed_t[s].shape == (n_mv,)

    def test_nonzero_everywhere(self, ground_config):
        """The bar never collapses to zero -- it's >= 1/2 across the grid."""
        r = SNRCalculator(ground_config).sweep()
        assert np.all(r.snr_band_unit["combined"] >= 0.5 - 1e-9)
        assert np.all(r.snr_band_unit["combined"] <= 1.0 + 1e-9)

    def test_band_independent_of_coadding(self, ground_config):
        """Regression: the coadded-SNR bar must not carry a sqrt(num_coadds)."""
        c1 = ground_config.model_copy(deep=True)
        c1.observation.num_frames = 1
        c9 = ground_config.model_copy(deep=True)
        c9.observation.num_frames = 9
        r1 = SNRCalculator(c1).sweep()
        r9 = SNRCalculator(c9).sweep()
        # SNR scales by sqrt(9) = 3 ...
        assert np.allclose(r9.snr_grid, 3.0 * r1.snr_grid, rtol=1e-6)
        # ... but the bar half-widths are unchanged.
        for s in (*NOISE_SOURCES, "combined"):
            assert np.allclose(r9.snr_band_unit[s], r1.snr_band_unit[s],
                               rtol=1e-9, atol=1e-12)

    def test_read_falls_and_sky_rises_with_exposure(self, ground_config):
        """At a faint magnitude, read's share drops and sky's grows with t."""
        r = SNRCalculator(ground_config).sweep()
        i = len(r.mvs) - 1  # faintest
        read = r.snr_band_unit["read"][i]
        sky = r.snr_band_unit["sky_background"][i]
        assert read[0] > read[-1]
        assert sky[-1] > sky[0]

    def test_regime_crossover(self, ground_config):
        """With appreciable read noise, read leads at short t and sky at long t."""
        cfg = ground_config.model_copy(deep=True)
        cfg.detector.read_noise = 5.0
        r = SNRCalculator(cfg).sweep()
        i = len(r.mvs) - 1
        read = r.snr_band_unit["read"][i]
        sky = r.snr_band_unit["sky_background"][i]
        assert read[0] > sky[0]      # read leads at short exposure
        assert sky[-1] > read[-1]    # sky leads at long exposure

    def test_shot_dominates_when_bright(self, ground_config):
        """A bright target with a long exposure is target-shot (photon) limited."""
        r = SNRCalculator(ground_config).sweep()
        i, j = 0, -1  # brightest magnitude, longest exposure
        contribs = {s: r.snr_band_unit[s][i, j] for s in NOISE_SOURCES}
        # target shot is the dominant contribution and pulls the bar toward 1/2
        assert max(contribs, key=contribs.get) == "target_shot"
        assert r.snr_band_unit["combined"][i, j] < 0.75

    def test_brighter_sky_grows_combined(self, ground_config):
        """Lowering sky_bkg_mv (brighter sky) widens the combined bar."""
        dark = ground_config.model_copy(deep=True)
        dark.atmosphere.sky_bkg_mv = 22.0
        bright = ground_config.model_copy(deep=True)
        bright.atmosphere.sky_bkg_mv = 18.0
        rd = SNRCalculator(dark).sweep()
        rb = SNRCalculator(bright).sweep()
        i, j = len(rd.mvs) - 1, -1  # faint, long exposure
        assert rb.snr_band_unit["combined"][i, j] > rd.snr_band_unit["combined"][i, j]
        assert rb.snr_band_unit["sky_background"][i, j] > rd.snr_band_unit["sky_background"][i, j]


# ── Search-rate envelope ─────────────────────────────────────────────────────

class TestSearchRateBand:

    def test_band_brackets_nominal(self, ground_config):
        """lo <= nominal <= hi for the precomputed combined and per-source bands."""
        ground_config.observation.snr_threshold = 3.0
        ground_config.output.visualize_noise = True  # precompute per-source bands
        r = SNRCalculator(ground_config).sweep()
        for s in ["combined", "read", "sky_background"]:
            lo, hi = r.search_rate_band[s]
            assert np.all(hi >= r.search_rates - 1e-9)
            assert np.all(lo <= r.search_rates + 1e-9)
            assert np.all(hi >= lo - 1e-9)

    def test_wider_band_widens_envelope(self, ground_config):
        """A larger sigma can only widen (never shrink) the rate envelope."""
        c1 = ground_config.model_copy(deep=True)
        c1.observation.snr_threshold = 3.0
        c1.output.sigma = 1.0
        c3 = ground_config.model_copy(deep=True)
        c3.observation.snr_threshold = 3.0
        c3.output.sigma = 3.0
        lo1, hi1 = SNRCalculator(c1).sweep().search_rate_band["combined"]
        lo3, hi3 = SNRCalculator(c3).sweep().search_rate_band["combined"]
        assert np.all(hi3 >= hi1 - 1e-9)
        assert np.all(lo3 <= lo1 + 1e-9)

    def test_band_absent_when_sigma_zero(self, ground_config):
        """No band is precomputed when the SNR band is disabled."""
        ground_config.output.sigma = 0.0
        r = SNRCalculator(ground_config).sweep()
        assert r.search_rate_band == {}


# ── Config + plotting integration ────────────────────────────────────────────

class TestConfigAndPlotting:

    def test_output_defaults(self):
        o = OutputConfig()
        assert o.sigma == 1.0
        assert o.visualize_noise is False

    def test_negative_sigma_rejected(self):
        with pytest.raises(Exception):
            OutputConfig(sigma=-1.0)

    @pytest.mark.parametrize("sigma,viz", [
        (1.0, False),   # default: single combined bar
        (1.0, True),    # combined bar + per-source overlay
        (0.0, False),   # bar disabled
        (2.5, True),
    ])
    def test_plot_summary_runs(self, ground_config, sigma, viz):
        ground_config.output.sigma = sigma
        ground_config.output.visualize_noise = viz
        r = SNRCalculator(ground_config).sweep()
        fig = plot_summary(ground_config, r)
        plt.close(fig)
