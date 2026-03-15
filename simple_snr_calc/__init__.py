"""simple_snr_calc - Signal-to-noise ratio calculator for satellite remote sensing."""

from .config import load_config, SNRConfig
from .snr import SNRCalculator, SNRResult, SweepResult
from .noise import NoiseBudget, compute_noise
from .spectrum import (
    load_solar_radiance,
    planck_spectral_radiance,
    load_curve,
    watts_to_photons,
    resample_to_grid,
)
from .target import (
    phase_angle_factor,
    radius_from_mv,
    solar_irradiance_at_target,
    thermal_photon_rate,
)
from .atmosphere import (
    sky_background_power,
    seeing_sigma_rad,
    combined_psf_sigma_rad,
)
from .optics import (
    build_throughput,
    compute_ifov,
    compute_fov,
    compute_ensquared_energy,
    peak_pixel_fraction,
    compute_zero_point,
    watts_to_photons_factor,
)
from .search_rate import compute_search_rates
from .plotting import plot_summary

__all__ = [
    # Core API
    "load_config",
    "SNRConfig",
    "SNRCalculator",
    "SNRResult",
    "SweepResult",
    "NoiseBudget",
    "compute_noise",
    # Spectrum / data
    "load_solar_radiance",
    "planck_spectral_radiance",
    "load_curve",
    "watts_to_photons",
    "resample_to_grid",
    # Target
    "phase_angle_factor",
    "radius_from_mv",
    "solar_irradiance_at_target",
    "thermal_photon_rate",
    # Atmosphere
    "sky_background_power",
    "seeing_sigma_rad",
    "combined_psf_sigma_rad",
    # Optics
    "build_throughput",
    "compute_ifov",
    "compute_fov",
    "compute_ensquared_energy",
    "peak_pixel_fraction",
    "compute_zero_point",
    "watts_to_photons_factor",
    # Search rate
    "compute_search_rates",
    # Plotting
    "plot_summary",
]
