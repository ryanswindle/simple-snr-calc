"""Search rate calculation.

The search rate at a given magnitude is set by the *crossing exposure*: the
shortest integration whose SNR reaches the detection threshold. Rather than read
that crossing off a fixed exposure grid -- which caps at the grid's longest
exposure and cliff-drops the rate to 0 once a faint star needs longer than that
-- we root-find the crossing per magnitude with a geometric (log-spaced)
bisection. That stays accurate across many decades of exposure, so the rate
declines smoothly toward 0 at faint magnitudes instead of falling off a cliff.

Everything here works through a ``snr_at(mv, t) -> SNR`` callable, so the same
crossing logic serves the model sweep, the on-sky overlay, and the SNR error
band (which just perturbs ``snr_at`` up/down before root-finding).
"""

import math

import numpy as np


def exposure_to_reach(snr_at, mv: float, target_snr: float,
                      t_lo: float, t_hi: float, n_iter: int = 40) -> float | None:
    """Smallest exposure in ``[t_lo, t_hi]`` whose ``snr_at(mv, t) >= target``.

    Returns ``t_lo`` if the target is already met there, ``None`` if it is
    unreachable by ``t_hi``, else the crossing exposure found by geometric
    bisection. SNR rises monotonically with exposure (true even with the
    signal-proportional scintillation term -- its contribution cancels out of
    d(SNR)/dt), so the bisection converges; the geometric midpoint keeps it
    accurate whether the crossing is at 0.3 s or 30000 s.
    """
    if snr_at(mv, t_lo) >= target_snr:
        return t_lo
    if snr_at(mv, t_hi) < target_snr:
        return None
    lo, hi = t_lo, t_hi
    for _ in range(n_iter):
        mid = math.sqrt(lo * hi)
        if snr_at(mv, mid) >= target_snr:
            hi = mid
        else:
            lo = mid
    return hi


def auto_max_exposure(snr_at, faint_mv: float, target_snr: float,
                      t_lo: float, hard_cap: float = 1.0e6) -> float:
    """Exposure that just reaches ``faint_mv`` -- the auto search-rate cap.

    Capping the per-magnitude root-find at this value lets the search-rate curve
    run smoothly down to ~0 right at the faint magnitude edge with no cliff. If
    ``faint_mv`` is unreachable even at ``hard_cap``, returns ``hard_cap``.
    """
    t_edge = exposure_to_reach(snr_at, float(faint_mv), target_snr, t_lo, hard_cap)
    return t_edge if t_edge is not None else hard_cap


def compute_search_rates(snr_at, mvs: np.ndarray, target_snr: float,
                         fov: tuple[float, float], obs_config,
                         frame_rate: float | None = None,
                         step_settle_time: float = 5.0,
                         t_lo: float = 0.1,
                         max_exposure_s: float | None = None) -> np.ndarray:
    """Search rate (deg^2/hr) per magnitude from a ``snr_at(mv, t)`` callable.

    For each magnitude the crossing exposure ``t_cross`` is root-found (geometric
    bisection, :func:`exposure_to_reach`); the rate is
    ``num_fields * fov / ((t_cross * num_frames + duty) / 3600)`` with
    ``duty = step_settle_time + readout``. A magnitude whose target is
    unreachable within ``max_exposure_s`` gets rate 0.

    Parameters
    ----------
    snr_at : callable
        ``snr_at(mv, t) -> SNR`` for apparent magnitude ``mv`` and exposure ``t``.
    mvs : ndarray
        Apparent magnitudes, ascending (the faintest, ``mvs[-1]``, sets the auto
        exposure cap).
    target_snr : float
        Detection threshold the SNR must reach.
    fov : tuple
        ``(fov_h_deg, fov_w_deg)``.
    obs_config : ObservationConfig
        Provides ``num_frames`` (coadds per field) and ``num_fields``.
    frame_rate : float or None
        Detector frame rate in Hz (sets readout time).
    step_settle_time : float
        Slew + settle time between fields, seconds.
    t_lo : float
        Shortest exposure considered (lower bisection bound).
    max_exposure_s : float or None
        Longest exposure the root-find may use. ``None`` auto-selects the
        exposure that just reaches ``mvs[-1]`` (:func:`auto_max_exposure`), so the
        rate tapers to ~0 at the faint edge instead of cliff-dropping.
    """
    fov_h, fov_w = fov
    num_frames = obs_config.num_frames
    num_fields = obs_config.num_fields
    readout_time = 1.0 / frame_rate if frame_rate else 0.0
    duty_time = step_settle_time + readout_time

    if max_exposure_s is None:
        max_exposure_s = auto_max_exposure(snr_at, float(mvs[-1]), target_snr, t_lo)

    search_rates = np.zeros(len(mvs))
    for i, mv in enumerate(mvs):
        t_cross = exposure_to_reach(snr_at, float(mv), target_snr,
                                    t_lo, max_exposure_s)
        if t_cross is None:
            continue
        cadence_s = t_cross * num_frames + duty_time
        search_rates[i] = num_fields * fov_h * fov_w / (cadence_s / 3600.0)

    return search_rates


def compute_search_rate_band(snr_band_at, mvs: np.ndarray,
                             target_snr: float, fov: tuple[float, float],
                             obs_config, frame_rate: float | None = None,
                             step_settle_time: float = 5.0,
                             t_lo: float = 0.1,
                             max_exposure_s: float | None = None,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Search-rate envelope implied by an SNR error band.

    The rate is driven by when each magnitude's SNR-vs-time curve crosses the
    threshold. Perturbing that curve up/down by the SNR band half-width moves the
    crossing and hence the rate: a higher SNR reaches threshold sooner, so
    ``snr + half-width`` yields the upper (faster) rate and ``snr - half-width``
    the lower one. Both perturbations share the *nominal* exposure cap, so the
    envelope is computed on the same exposure budget as the nominal rate.

    Parameters
    ----------
    snr_band_at : callable
        ``snr_band_at(mv, t) -> (snr, half_width)`` -- the nominal SNR and the
        band half-width in SNR units (already scaled by ``output.sigma``), both
        from a single evaluation so the crossing root-find costs one SNR call per
        step rather than two.

    Returns
    -------
    (lo, hi) : tuple of ndarray
        Lower and upper search rates in deg^2/hr for each magnitude.
    """
    if max_exposure_s is None:
        max_exposure_s = auto_max_exposure(
            lambda mv, t: snr_band_at(mv, t)[0], float(mvs[-1]), target_snr, t_lo)

    def hi_at(mv, t):
        snr, hw = snr_band_at(mv, t)
        return snr + hw

    def lo_at(mv, t):
        snr, hw = snr_band_at(mv, t)
        return max(snr - hw, 0.0)

    hi = compute_search_rates(hi_at, mvs, target_snr, fov, obs_config,
                              frame_rate, step_settle_time, t_lo, max_exposure_s)
    lo = compute_search_rates(lo_at, mvs, target_snr, fov, obs_config,
                              frame_rate, step_settle_time, t_lo, max_exposure_s)
    return lo, hi
