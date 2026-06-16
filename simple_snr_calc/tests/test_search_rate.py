"""Tests for the root-find search-rate path.

The search rate is set by the crossing exposure -- the shortest integration whose
SNR reaches the detection threshold -- root-found per magnitude by geometric
bisection (:func:`exposure_to_reach`). This replaces reading the crossing off a
fixed linear exposure grid, which capped at the grid's longest exposure and made
the rate cliff-drop to 0 once a faint star needed longer than that. With an
auto-selected exposure cap that just reaches the faintest magnitude, the rate now
tapers smoothly to ~0 at the faint edge instead.
"""

import numpy as np
import pytest

from simple_snr_calc.noise import snr_band_unit
from simple_snr_calc.search_rate import (
    auto_max_exposure,
    compute_search_rate_band,
    compute_search_rates,
    exposure_to_reach,
)
from simple_snr_calc.snr import SNRCalculator


# ── exposure_to_reach: geometric bisection across decades ────────────────────

class TestExposureToReach:

    def test_finds_crossing_across_decades(self):
        """A crossing far up the exposure range is found accurately."""
        for t_star in [0.3, 12.0, 1234.5, 87654.0]:
            snr_at = lambda mv, t, ts=t_star: 6.0 * t / ts  # monotonic in t
            t = exposure_to_reach(snr_at, 0.0, 6.0, 0.1, 1.0e6)
            assert t == pytest.approx(t_star, rel=1e-3)

    def test_returns_t_lo_when_already_met(self):
        snr_at = lambda mv, t: 100.0
        assert exposure_to_reach(snr_at, 0.0, 6.0, 0.1, 1.0e6) == 0.1

    def test_returns_none_when_unreachable(self):
        snr_at = lambda mv, t: 1.0
        assert exposure_to_reach(snr_at, 0.0, 6.0, 0.1, 1.0e6) is None

    def test_auto_max_exposure_matches_crossing(self):
        snr_at = lambda mv, t: 6.0 * t / 5000.0
        assert auto_max_exposure(snr_at, 0.0, 6.0, 0.1) == pytest.approx(5000.0,
                                                                         rel=1e-3)

    def test_auto_max_exposure_falls_back_to_hard_cap(self):
        snr_at = lambda mv, t: 1.0  # never reaches threshold
        assert auto_max_exposure(snr_at, 0.0, 6.0, 0.1, hard_cap=1234.0) == 1234.0


# ── The smooth falloff on a real sweep ───────────────────────────────────────

class TestSmoothFalloff:

    def test_auto_cap_reaches_faint_edge_no_cliff(self, ground_config):
        """Auto cap makes every magnitude reachable -> no cliff to exactly 0."""
        r = SNRCalculator(ground_config).sweep()
        assert np.all(r.search_rates > 0)

    def test_rate_monotonically_non_increasing(self, ground_config):
        """Brighter magnitudes survey faster; the curve never rises with mv."""
        r = SNRCalculator(ground_config).sweep()
        tol = 1e-6 * r.search_rates.max()
        assert np.all(np.diff(r.search_rates) <= tol)

    def test_cap_truncates_what_auto_reaches(self, ground_config):
        """A finite cap drops the faint tail to 0 without changing reachable mags."""
        auto = SNRCalculator(ground_config).sweep()
        capped_cfg = ground_config.model_copy(deep=True)
        capped_cfg.observation.max_search_exposure_s = 10.0
        cap = SNRCalculator(capped_cfg).sweep()

        assert np.all(auto.search_rates > 0)       # auto reaches the faint edge
        assert cap.search_rates[-1] == 0.0         # 10 s can't reach the faintest
        assert np.count_nonzero(cap.search_rates == 0) > 0
        # The zero tail is a clean faint-end truncation (contiguous nonzero head).
        nz = np.nonzero(cap.search_rates > 0)[0]
        assert np.all(np.diff(nz) == 1)
        # Where both are nonzero the crossing is identical (cap is not binding).
        both = (auto.search_rates > 0) & (cap.search_rates > 0)
        assert np.allclose(auto.search_rates[both], cap.search_rates[both],
                           rtol=1e-6)

    def test_brighter_target_surveys_faster(self, ground_config):
        """Sanity: the bright end has a strictly higher rate than the faint end."""
        r = SNRCalculator(ground_config).sweep()
        assert r.search_rates[0] > r.search_rates[-1]


# ── compute_search_rate_band: the function-level contract ────────────────────

class TestSearchRateBandFunction:

    def _callables(self, calc, sigma=1.0, key="combined"):
        def snr_at(mv, t):
            return calc.compute_snr(mv, t).snr

        def snr_band_at(mv, t):
            res = calc.compute_snr(mv, t)
            return res.snr, sigma * snr_band_unit(res.noise)[key]

        return snr_at, snr_band_at

    def test_envelope_brackets_nominal(self, ground_config):
        ground_config.observation.snr_threshold = 3.0
        calc = SNRCalculator(ground_config)
        obs = ground_config.observation
        mvs = np.arange(obs.mv_range[0], obs.mv_range[1] + 0.5, 1.0)
        snr_at, snr_band_at = self._callables(calc)
        nominal = compute_search_rates(
            snr_at, mvs, obs.snr_threshold, calc.fov, obs,
            ground_config.detector.frame_rate,
            step_settle_time=ground_config.optics.step_settle_time,
            t_lo=obs.exposure_range[0])
        lo, hi = compute_search_rate_band(
            snr_band_at, mvs, obs.snr_threshold, calc.fov, obs,
            ground_config.detector.frame_rate,
            step_settle_time=ground_config.optics.step_settle_time,
            t_lo=obs.exposure_range[0])
        assert np.all(hi >= nominal - 1e-9)
        assert np.all(lo <= nominal + 1e-9)
        assert np.all(hi >= lo - 1e-9)
