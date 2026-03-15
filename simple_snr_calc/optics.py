"""Optical system: throughput chain, ensquared energy, peak pixel fraction, FOV.

Peak Pixel Fraction
====================

The SNR calculation needs the signal in a single pixel (or binned pixel),
because that's what we compare against per-pixel noise sources (read noise,
dark current, sky background). But the target's PSF spreads light across
multiple pixels. The peak pixel fraction answers: what fraction of the total
signal lands in the brightest pixel?

    pixel_signal = total_photons * peak_pixel_fraction
    SNR = pixel_signal / sqrt(pixel_signal + sky + dark + read^2 + quant^2)

The model assumes a 2D Gaussian PSF with standard deviation sigma (in pixel
units, derived from atmosphere + jitter), convolved with a uniform streak of
length L pixels (L = streak_rate * exposure_time). The source lands at a
random sub-pixel position. We compute the expected signal fraction in the
brightest pixel by averaging over sub-pixel offsets.

Step 1 -- sigma from ensquared energy (EE):

    The centered EE for a 2D Gaussian over a square pixel of width 1 is:

        EE = [erf(a / (sigma * sqrt(2)))]^2

    where a = 0.5 (half-pixel) and erf is the Gauss error function:

        erf(x) = (2 / sqrt(pi)) * integral from 0 to x of exp(-t^2) dt

    Inverting gives sigma from EE:

        sigma = 0.5 / (sqrt(2) * erfinv(sqrt(EE)))

Step 2 -- signal fraction in pixel [k, k+1] for source at offset (dx, dy):

    Cross-streak (y-axis), pure Gaussian:

        f_y = 0.5 * [erf((1 - dy) / (sigma * sqrt(2)))
                    - erf(-dy / (sigma * sqrt(2)))]

    Along-streak (x-axis), Gaussian convolved with boxcar of length L:

        f_x(k) = (1/L) * integral from -L/2 to L/2 of
                  0.5 * [erf((k + 1 - dx - t) / (sigma * sqrt(2)))
                       - erf((k - dx - t) / (sigma * sqrt(2)))] dt

    where t runs along the streak and L is the streak length in pixels.

Step 3 -- peak pixel: for each sub-pixel offset, the brightest pixel has:

    f_best(dx, dy) = max_k[f_x(k)] * f_y

Step 4 -- average over sub-pixel offsets on a uniform grid:

    peak_pixel_fraction = (1/N^2) * sum over (dx, dy) of f_best(dx, dy)

Special case (L = 0, rate-track mode): no streak, so f_x has the same form
as f_y, and the result reduces to the sub-pixel-averaged ensquared energy. This is
slightly less than the centered EE because most random sub-pixel positions
are not perfectly centered on the pixel.
"""

import numpy as np
from scipy.special import erf, erfinv

from ._constants import SUN_MV
from .config import resolve_data_path
from .spectrum import load_curve, resample_to_grid


def build_throughput(wl_grid: np.ndarray, optics_files: list[str],
                     filter_name: str | None, filter_file: str | None,
                     qe_file: str) -> np.ndarray:
    """Build combined system throughput on the given wavelength grid.

    Multiplies optics coatings x filter x QE curves together.

    Parameters
    ----------
    wl_grid : ndarray
        Target wavelength grid in nm.
    optics_files : list of str
        Paths to optics throughput CSVs (relative to data dir).
    filter_name : str or None
        Filter name (looked up in data/filters/).
    filter_file : str or None
        Explicit filter file path (overrides filter_name).
    qe_file : str
        QE curve file path (relative to data dir).

    Returns
    -------
    throughput : ndarray
        Combined throughput on wl_grid.
    """
    throughput = np.ones_like(wl_grid)

    # Optics (mirrors, lenses) — one file per surface
    for f in optics_files:
        wl, values = load_curve(resolve_data_path(f))
        resampled = resample_to_grid(wl, values, wl_grid)
        throughput *= np.clip(resampled, 0.0, 1.0)

    # Filter
    fpath = None
    if filter_file:
        fpath = resolve_data_path(filter_file)
    elif filter_name:
        fpath = resolve_data_path(f"filters/{filter_name}.csv")

    if fpath is not None:
        wl, values = load_curve(fpath)
        resampled = resample_to_grid(wl, values, wl_grid)
        throughput *= np.clip(resampled, 0.0, 1.0)

    # QE
    wl, values = load_curve(resolve_data_path(qe_file))
    resampled = resample_to_grid(wl, values, wl_grid)
    throughput *= np.clip(resampled, 0.0, 1.0)

    return throughput


def compute_ifov(pixel_size: float, focal_length: float) -> float:
    """Compute instantaneous field of view in arcsec.

    Parameters
    ----------
    pixel_size : float
        Pixel size in meters.
    focal_length : float
        Focal length in meters.
    """
    return float(2.0 * np.arctan(pixel_size / (2.0 * focal_length)) * 206265.0)


def compute_fov(ifov_arcsec: float, width_pix: int, height_pix: int,
                image_circle_mm: float | None = None,
                pixel_size_m: float | None = None) -> tuple[float, float]:
    """Compute effective FOV in degrees, accounting for image circle.

    Returns (fov_h_deg, fov_w_deg).
    """
    fov_h = ifov_arcsec * height_pix / 3600.0
    fov_w = ifov_arcsec * width_pix / 3600.0

    if image_circle_mm is not None and pixel_size_m is not None:
        sensor_w_mm = width_pix * pixel_size_m * 1e3
        sensor_h_mm = height_pix * pixel_size_m * 1e3
        sensor_diag_mm = np.sqrt(sensor_w_mm ** 2 + sensor_h_mm ** 2)

        if image_circle_mm < sensor_w_mm:
            # Image circle fits inside sensor
            ic_deg = image_circle_mm * 1e-3 * ifov_arcsec / pixel_size_m / 3600.0
            ic_area = np.pi * ic_deg ** 2 / 4.0
            fov_h = np.sqrt(ic_area)
            fov_w = np.sqrt(ic_area)
        elif sensor_diag_mm > image_circle_mm:
            # Partial overlap — compute intersection area
            w = fov_w
            h = fov_h
            r = image_circle_mm / 2.0 * 1e-3 * ifov_arcsec / pixel_size_m / 3600.0
            area = (w * h
                    - 2.0 * w * np.sqrt(r ** 2 - h ** 2 / 4.0)
                    - 2.0 * h * np.sqrt(r ** 2 - w ** 2 / 4.0)
                    + 4.0 * r ** 2 * np.arcsin(w / (2.0 * r))
                    + 4.0 * r ** 2 * np.arcsin(h / (2.0 * r))
                    - 2.0 * np.pi * r ** 2)
            fov_h = np.sqrt(abs(area))
            fov_w = np.sqrt(abs(area))

    return float(fov_h), float(fov_w)


def compute_ensquared_energy(psf_sigma_rad: float, ifov_arcsec: float,
                             binning: int = 1) -> tuple[float, float]:
    """Ensquared energy for a Gaussian PSF centered on a square pixel.

    Uses the analytical result: EE = [erf(a / (sigma * sqrt(2)))]^2
    where a = half-pixel angular size in radians.

    Parameters
    ----------
    psf_sigma_rad : float
        Combined PSF standard deviation in radians.
    ifov_arcsec : float
        Pixel IFOV in arcsec (unbinned).
    binning : int
        Symmetric binning factor.

    Returns
    -------
    ee : float
        Ensquared energy fraction.
    psf_sigma_rad : float
        The input PSF sigma (passed through for convenience).
    """
    theta_pixel_rad = ifov_arcsec * 4.848e-6 * binning
    half_pixel = theta_pixel_rad / 2.0
    ee = float(erf(half_pixel / (psf_sigma_rad * np.sqrt(2.0))) ** 2)
    return ee, psf_sigma_rad


def _sigma_from_ee(ee: float, half_pixel: float = 0.5) -> float:
    """Compute Gaussian PSF sigma from ensquared energy.

    Inverts EE = erf(a / (sigma * sqrt(2)))^2 to get:
    sigma = a / (sqrt(2) * erfinv(sqrt(ee)))

    Parameters
    ----------
    ee : float
        Ensquared energy (0, 1).
    half_pixel : float
        Half-pixel size in arbitrary units (default 0.5 for unit pixels).
    """
    return half_pixel / (np.sqrt(2.0) * erfinv(np.sqrt(ee)))


def peak_pixel_fraction(ensquared_energy: float,
                            streak_length_pix: float,
                            n_sub: int = 100) -> float:
    """Compute the peak pixel fraction analytically.

    For a Gaussian PSF (with sigma derived from ensquared_energy) convolved
    with a uniform streak of the given length, computes the expected fraction
    of total signal in the brightest pixel, averaged over random sub-pixel
    offsets.

    Parameters
    ----------
    ensquared_energy : float
        Fraction of PSF energy in one pixel (centered case).
    streak_length_pix : float
        Streak length in pixels (0 for rate-track mode).
    n_sub : int
        Number of sub-pixel offset samples per axis (n_sub^2 total).
    """
    sigma = _sigma_from_ee(ensquared_energy)
    s2 = sigma * np.sqrt(2.0)
    L = max(streak_length_pix, 0.0)

    # Sub-pixel offset grid
    offsets = np.linspace(0, 1, n_sub, endpoint=False) + 0.5 / n_sub
    dx_grid, dy_grid = np.meshgrid(offsets, offsets)
    dx = dx_grid.ravel()
    dy = dy_grid.ravel()

    # Cross-streak (y-axis): Gaussian fraction in peak pixel
    # Source at position dy within pixel [0, 1)
    y_frac = 0.5 * (erf((1.0 - dy) / s2) - erf(-dy / s2))

    if L <= 1e-12:
        # No streak: x-axis same as y-axis
        x_frac = 0.5 * (erf((1.0 - dx) / s2) - erf(-dx / s2))
        return float(np.mean(x_frac * y_frac))

    # Along-streak: Gaussian convolved with boxcar of length L
    # Check candidate pixels to find the brightest one
    k_min = int(np.floor(-L / 2)) - 1
    k_max = int(np.ceil(L / 2)) + 2

    n_streak = max(int(L * 50), 200)
    t = np.linspace(-L / 2, L / 2, n_streak)

    best_x = np.zeros(len(dx))
    for k in range(k_min, k_max + 1):
        # Fraction of streaked PSF signal in pixel [k, k+1]
        integrand = 0.5 * (erf((k + 1 - dx[:, None] - t[None, :]) / s2)
                         - erf((k - dx[:, None] - t[None, :]) / s2))
        x_frac = np.trapezoid(integrand, t, axis=1) / L
        best_x = np.maximum(best_x, x_frac)

    return float(np.mean(best_x * y_frac))


def compute_zero_point(wl_nm: np.ndarray, solar_irr_photons: np.ndarray,
                       throughput: np.ndarray,
                       aperture_area: float) -> float:
    """Compute the system zero point magnitude.

    The zero point is the magnitude at which a solar-spectrum source
    produces a total detected photon rate equal to the integrated
    solar flux through the system.

    Parameters
    ----------
    wl_nm : ndarray
        Wavelength grid in nm.
    solar_irr_photons : ndarray
        Solar irradiance at 1 AU in photons/s/m^2/nm.
    throughput : ndarray
        System throughput.
    aperture_area : float
        Effective collecting area in m^2.

    Returns
    -------
    zero_point : float
        System zero point magnitude.
    """
    flux_total = np.trapezoid(solar_irr_photons * aperture_area * throughput, wl_nm)
    return SUN_MV + 2.5 * np.log10(flux_total)


def watts_to_photons_factor(wl_nm: np.ndarray,
                            solar_irr_watts: np.ndarray,
                            throughput: np.ndarray) -> float:
    """Spectral-weighted conversion: watts at aperture -> photons/s at FPA.

    eta = integral[E(lam) * T(lam) * lam/(hc) dlam] / integral[E(lam) dlam]

    This converts broadband watts (weighted by solar spectrum) to detected
    photons/s through the optical system.
    """
    from astropy import constants as const
    h = const.h.value
    c = const.c.value

    # Restrict to wavelengths where throughput is nonzero, so the
    # denominator isn't diluted by far-IR tail with zero response.
    mask = throughput > 1e-6
    wl_m = wl_nm[mask] * 1e-9
    numerator = np.trapezoid(solar_irr_watts[mask] * throughput[mask] * wl_m / (h * c), wl_nm[mask])
    denominator = np.trapezoid(solar_irr_watts[mask], wl_nm[mask])
    return numerator / denominator
