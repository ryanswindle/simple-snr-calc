"""Search rate calculation."""

import numpy as np


def compute_search_rates(mvs: np.ndarray, snr_grid: np.ndarray,
                         exposure_times: np.ndarray,
                         fov: tuple[float, float],
                         obs_config, frame_rate: float | None = None,
                         step_settle_time: float = 5.0) -> np.ndarray:
    """Compute search rate for each magnitude.

    Parameters
    ----------
    mvs : ndarray
        Apparent magnitudes.
    snr_grid : 2D ndarray
        SNR[i, j] for mvs[i], exposure_times[j].
    exposure_times : ndarray
        Exposure times in seconds.
    fov : tuple
        (fov_h_deg, fov_w_deg).
    obs_config : ObservationConfig
        Observation configuration (num_frames, num_fields, snr_threshold).
    frame_rate : float or None
        Detector frame rate in Hz.
    step_settle_time : float
        Time to slew and settle between fields, in seconds.

    Returns
    -------
    search_rates : ndarray
        Search rate in deg^2/hr for each magnitude.
    """
    fov_h, fov_w = fov
    snr_threshold = obs_config.snr_threshold
    num_frames = obs_config.num_frames
    num_fields = obs_config.num_fields

    readout_time = 1.0 / frame_rate if frame_rate else 0.0

    search_rates = np.zeros(len(mvs))
    for i in range(len(mvs)):
        snr_vs_t = snr_grid[i, :]
        if np.max(snr_vs_t) >= snr_threshold:
            idx = int(np.argmax(snr_vs_t >= snr_threshold))
            # Interpolate exact crossing time between grid points
            if idx > 0 and snr_vs_t[idx - 1] < snr_threshold:
                # Linear interpolation between idx-1 and idx
                frac = ((snr_threshold - snr_vs_t[idx - 1])
                        / (snr_vs_t[idx] - snr_vs_t[idx - 1]))
                t_cross = (exposure_times[idx - 1]
                           + frac * (exposure_times[idx] - exposure_times[idx - 1]))
            else:
                t_cross = exposure_times[idx]
            t_snr = t_cross * num_frames
            duty_time = max(step_settle_time, readout_time)
            search_rates[i] = (
                num_fields * fov_h * fov_w / ((t_snr + duty_time) / 3600.0)
            )

    return search_rates
