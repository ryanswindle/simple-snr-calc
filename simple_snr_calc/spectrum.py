"""Solar spectrum, Planck blackbody, and spectral utilities."""

import numpy as np
from functools import lru_cache
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from astropy import constants as const

from .config import resolve_data_path

# Physical constants
_h = const.h.value       # J·s
_c = const.c.value       # m/s
_k_B = const.k_B.value   # J/K


@lru_cache(maxsize=1)
def load_solar_radiance(filename: str = "solar/solar_radiance.csv"):
    """Load solar spectral radiance from CSV.
    Obtained from ASTM_E490_00A_2006, "Standard Solar Constant and Zero Air Mass Solar Spectral Irradiance Tables"

    Returns
    -------
    wl_nm : ndarray
        Wavelengths in nm.
    radiance : ndarray
        Spectral radiance in W/m^2/sr/nm at the Sun's surface.
    """
    path = resolve_data_path(filename)
    data = np.genfromtxt(path, delimiter=',', skip_header=3)
    return data[:, 0].copy(), data[:, 1].copy()


def watts_to_photons(wl_nm: np.ndarray, radiance_watts: np.ndarray) -> np.ndarray:
    """Convert spectral radiance from W/m^2/sr/nm to photons/s/m^2/sr/nm."""
    return radiance_watts * wl_nm * 1e-9 / (_h * _c)


def planck_spectral_radiance(wl_nm: np.ndarray, temperature_K: float) -> np.ndarray:
    """Planck spectral radiance B(lambda, T) in W/m^2/sr/nm.

    Parameters
    ----------
    wl_nm : ndarray
        Wavelengths in nm.
    temperature_K : float
        Temperature in Kelvin.
    """
    wl_m = wl_nm * 1e-9
    a = 2.0 * _h * _c**2 / wl_m**5
    b = np.expm1(_h * _c / (wl_m * _k_B * temperature_K))
    # Convert W/m^2/sr/m -> W/m^2/sr/nm
    return a / b * 1e-9


def load_curve(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column CSV curve (wavelength nm, value).

    Handles files with or without a header row.
    """
    path = Path(filepath)
    data = np.genfromtxt(path, delimiter=',', dtype=float)
    # Skip header rows that parsed as NaN
    mask = ~np.isnan(data[:, 0])
    data = data[mask]
    return data[:, 0].copy(), data[:, 1].copy()


def resample_to_grid(wl_source: np.ndarray, values: np.ndarray,
                     wl_target: np.ndarray) -> np.ndarray:
    """Resample a curve onto a target wavelength grid using PCHIP interpolation.

    Values outside the source range are clamped to the nearest edge value.
    """
    interp = PchipInterpolator(wl_source, values, extrapolate=True)
    result = interp(wl_target)
    result[wl_target < wl_source[0]] = values[0]
    result[wl_target > wl_source[-1]] = values[-1]
    return result
