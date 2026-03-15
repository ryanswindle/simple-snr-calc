"""Atmospheric effects: sky background, extinction (TBD), sky background."""

import numpy as np

from ._constants import SUN_MV, SUN_IRRADIANCE


def sky_background_power(sky_bkg_mv: float, aperture_diameter: float) -> float:
    """Sky background power at aperture per arcsec^2.

    Parameters
    ----------
    sky_bkg_mv : float
        Sky brightness in mag/arcsec^2.
    aperture_diameter : float
        Aperture diameter in meters.

    Returns
    -------
    power : float
        Watts per arcsec^2 collected by the aperture.
    """
    mag_factor = 10.0 ** (-0.4 * (sky_bkg_mv - SUN_MV))
    aperture_area = np.pi * (aperture_diameter / 2.0) ** 2
    return mag_factor * aperture_area * SUN_IRRADIANCE


def seeing_sigma_rad(r0: float, wavelength: float) -> float:
    """Atmospheric seeing PSF standard deviation in radians.

    Parameters
    ----------
    r0 : float
        Fried parameter in meters.
    wavelength : float
        Wavelength in meters (at which r0 is defined).
    """
    fwhm = 0.98 * wavelength / r0
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def combined_psf_sigma_rad(r0: float, wavelength: float,
                           jitter_arcsec: float) -> float:
    """Combined PSF sigma (RSS of seeing + jitter) in radians.

    Parameters
    ----------
    r0 : float
        Fried parameter in meters. Use very large value for space-based.
    wavelength : float
        Wavelength in meters.
    jitter_arcsec : float
        Mount jitter in arcsec RMS.
    """
    atm_sigma = seeing_sigma_rad(r0, wavelength)
    jitter_rad = jitter_arcsec * 4.848e-6
    return np.sqrt(atm_sigma**2 + jitter_rad**2)
