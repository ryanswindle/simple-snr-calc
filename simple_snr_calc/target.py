"""Target radiance: reflected solar and thermal emission."""

import numpy as np
from astropy import constants as const

from ._constants import SUN_MV
from .spectrum import planck_spectral_radiance, watts_to_photons

_sun_radius = const.R_sun.value   # m
_au = const.au.value               # m


def solar_irradiance_at_target(solar_radiance: np.ndarray) -> np.ndarray:
    """Convert solar surface radiance to irradiance at 1 AU.

    Parameters
    ----------
    solar_radiance : ndarray
        W/m^2/sr/nm (or photons/s/m^2/sr/nm) at the Sun's surface.

    Returns
    -------
    irradiance : ndarray
        W/m^2/nm (or photons/s/m^2/nm) at 1 AU.
    """
    solid_angle = np.pi * (_sun_radius / _au) ** 2
    return solar_radiance * solid_angle


def phase_angle_factor(phase_angles_deg) -> np.ndarray:
    """Lambertian sphere phase angle factor (Tousey 1957, Sussman 1958).

    For a Lambertian sphere, the reflected intensity as a fraction of the
    full-phase geometric cross-section scales by this factor.

    Parameters
    ----------
    phase_angles_deg : array-like
        Solar phase angles in degrees.

    Returns
    -------
    paf : ndarray
        Phase angle factor for each input angle.
    """
    phi = np.deg2rad(np.asarray(phase_angles_deg, dtype=float))
    return (2.0 / (3.0 * np.pi ** 2)) * (np.sin(phi) + (np.pi - phi) * np.cos(phi))


def radius_from_mv(mv: float, albedo: float, range_m: float,
                   phase_angle_deg: float) -> float:
    """Derive target radius [m] from apparent visual magnitude.

    r = R * sqrt(10^((m_sun - mv) / 2.5) / (albedo * paf * pi))

    Parameters
    ----------
    mv : float
        Apparent visual magnitude.
    albedo : float
        Target albedo.
    range_m : float
        Target range in meters.
    phase_angle_deg : float
        Solar phase angle in degrees.
    """
    paf = phase_angle_factor([phase_angle_deg])[0]
    mag_ratio = 10.0 ** ((SUN_MV - mv) / 2.5)
    r_squared = range_m ** 2 * mag_ratio / (albedo * paf * np.pi)
    return np.sqrt(np.abs(r_squared))


def thermal_photon_rate(wl_nm: np.ndarray, throughput: np.ndarray,
                        temperature: float, emissivity: float,
                        target_radius: float, target_range: float,
                        aperture_area: float) -> float:
    """Compute thermal emission photon rate at the detector.

    For a Lambertian sphere of radius r at range R, the intensity is
    I = epsilon * B(lambda, T) * pi * r^2 [W/sr], and the irradiance at the
    aperture is F = I / R^2. Multiplied by the aperture area gives power,
    then weighted by throughput and converted to photons.

    Parameters
    ----------
    wl_nm : ndarray
        Wavelength grid in nm.
    throughput : ndarray
        System throughput on the same grid.
    temperature : float
        Target temperature in K.
    emissivity : float
        Target emissivity (0-1).
    target_radius : float
        Target radius in meters.
    target_range : float
        Target range in meters.
    aperture_area : float
        Effective collecting area in m^2.

    Returns
    -------
    photon_rate : float
        Total thermal photons/s at the detector.
    """
    # Planck spectral radiance [W/m^2/sr/nm]
    B = planck_spectral_radiance(wl_nm, temperature)

    # Convert to photons/s/m^2/sr/nm
    B_photons = watts_to_photons(wl_nm, B)

    # Geometry: Lambertian sphere intensity = B * pi * r^2, irradiance = I/R^2
    geom = np.pi * target_radius ** 2 * aperture_area / target_range ** 2

    # Detected spectral photon rate
    spectral_rate = emissivity * B_photons * geom * throughput

    # Integrate over wavelength
    return float(np.trapezoid(spectral_rate, wl_nm))
