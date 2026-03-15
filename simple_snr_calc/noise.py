"""Noise budget computation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseBudget:
    """Noise components in electrons."""
    target_shot: float
    sky_background: float
    dark_current: float
    read: float
    quantization: float
    total: float

    def variance_fractions(self) -> dict[str, float]:
        """Return noise variance fractions (noise power budget).

        Each value is the fraction of total noise variance contributed
        by that source. Fractions sum to 1.0.
        """
        total_var = self.total ** 2
        if total_var == 0:
            return {}
        return {
            'target_shot': self.target_shot ** 2 / total_var,
            'sky_background': self.sky_background ** 2 / total_var,
            'dark_current': self.dark_current ** 2 / total_var,
            'read': self.read ** 2 / total_var,
            'quantization': self.quantization ** 2 / total_var,
        }


def compute_noise(peak_signal: float, sky_bkg_rate: float,
                  dark_current: float, read_noise: float,
                  full_well: float, bit_depth: int,
                  exposure_time: float, binning: int,
                  is_cmos: bool) -> NoiseBudget:
    """Compute noise budget for a single pixel observation.

    Parameters
    ----------
    peak_signal : float
        Signal electrons in the peak pixel (total, includes exposure time).
    sky_bkg_rate : float
        Sky background rate in e-/pix/s.
    dark_current : float
        Dark current in e-/pix/s.
    read_noise : float
        Read noise in e-/pix (single pixel).
    full_well : float
        Full well capacity in e-.
    bit_depth : int
        ADC bit depth.
    exposure_time : float
        Integration time in seconds.
    binning : int
        Symmetric binning factor (NxN).
    is_cmos : bool
        If True, scale read noise by binning (CMOS reads each pixel
        independently, so N^2 pixels contribute N^2 variance terms,
        giving RMS = read_noise * N).
    """
    target_shot = np.sqrt(max(peak_signal, 0.0))
    sky_shot = np.sqrt(max(sky_bkg_rate * exposure_time, 0.0))
    dark_shot = np.sqrt(max(dark_current * exposure_time, 0.0))

    # CMOS: digital binning sums N^2 pixels, each with independent read noise
    # CCD: on-chip binning, single readout
    rn = read_noise * binning if is_cmos else read_noise

    quant = full_well / 2 ** bit_depth / np.sqrt(12.0)

    total = np.sqrt(
        target_shot ** 2
        + sky_shot ** 2
        + dark_shot ** 2
        + rn ** 2
        + quant ** 2
    )

    return NoiseBudget(
        target_shot=target_shot,
        sky_background=sky_shot,
        dark_current=dark_shot,
        read=rn,
        quantization=quant,
        total=total,
    )
