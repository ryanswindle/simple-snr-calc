"""Atmospheric effects: sky background, extinction (TBD), sky background."""

import numpy as np

from ._constants import SUN_MV, SUN_IRRADIANCE

# Atmospheric scale height used by the scintillation model (Young 1967), in
# meters. The turbulent layers responsible for scintillation thin out with this
# e-folding height, so a higher observatory sees less of them.
SCINTILLATION_SCALE_HEIGHT = 8000.0


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


def scintillation_fraction(
    aperture_diameter: float,
    exposure_time: float,
    airmass: float = 1.0,
    observatory_altitude: float = 0.0,
    coeff: float = 1.5,
) -> float:
    """Fractional scintillation noise (RMS of relative source intensity).

    The Young (1967) approximation, in the modern Osborn et al. (2015) form:

        sigma**2 = 10e-6 * C_Y**2 * D**(-4/3) * X**3
                   * exp(-2 * h_obs / H) / (2 * t)

    with ``D`` the aperture diameter in **meters**, ``X`` the airmass, ``h_obs``
    the observatory altitude (m), ``H`` = 8000 m the atmospheric scale height,
    and ``t`` the exposure time (s). ``C_Y`` is the empirical Young coefficient
    (theoretical median ~1.5); it absorbs the site/altitude degeneracy, so the
    cleanest way to pin it is to fit this ceiling to bright-star photometry.

    Scintillation is a *multiplicative* (signal-proportional) noise: it adds
    ``sigma * S`` electrons to a source of ``S`` electrons, so unlike shot, sky
    or read noise it does not average down with brightness. It imposes a
    magnitude-independent SNR ceiling of ``1 / sigma`` that scales as
    ``sqrt(t)`` -- exactly the bright-end flattening and exposure-ordered fan-out
    seen in on-sky data and absent from a pure shot/sky/read budget.

    Returns 0.0 when disabled (``coeff``, ``t`` or ``D`` non-positive).
    """
    if coeff <= 0.0 or exposure_time <= 0.0 or aperture_diameter <= 0.0:
        return 0.0
    variance = (
        10.0e-6
        * coeff ** 2
        * aperture_diameter ** (-4.0 / 3.0)
        * airmass ** 3
        * np.exp(-2.0 * observatory_altitude / SCINTILLATION_SCALE_HEIGHT)
        / (2.0 * exposure_time)
    )
    return float(np.sqrt(variance))
