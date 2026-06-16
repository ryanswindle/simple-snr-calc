"""Noise budget computation."""

from dataclasses import dataclass
import numpy as np

# Independent variance components of the noise budget, in the order they are
# summed in quadrature to form ``NoiseBudget.total``. These are the sources
# shown as individual SNR error bands when ``output.visualize_noise`` is set.
NOISE_SOURCES = (
    "target_shot",
    "sky_background",
    "dark_current",
    "read",
    "quantization",
)


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


def snr_band_unit(noise: NoiseBudget) -> dict[str, float]:
    """SNR error-bar half-widths at sigma = 1, propagated through the SNR.

    Treating the measured signal as the random variable and letting the noise
    estimate track it (as real photometry does), the 1-sigma error bar on the
    *coadded* ``SNR = sqrt(Nc) * S / sqrt(S + B)`` works out to

        sigma(SNR) = (S/2 + B) / (S + B) = 1 - f_shot / 2 ,

    where ``S`` is the target-shot variance, ``B`` is every other variance
    (sky + dark + read + quantization), and ``f_shot = (target_shot/total)**2``
    is the target-shot variance fraction. This is the honest bar: it includes
    every noise source, is nonzero everywhere -- it runs from 1/2 when
    target-shot (photon) limited to 1 when read/sky/dark/quant limited -- and
    is independent of the number of coadds (the sqrt(Nc) boost cancels between
    the SNR and the correspondingly sharpened flux estimate).

    The per-source *contributions* to the bar add linearly (each is
    proportional to that source's variance, not its RMS):

        target_shot:  f_shot / 2
        source X:     f_X = (sigma_X / total)**2     (sky, dark, read, quant)

    Target shot is down-weighted by 1/2 because it is the only source
    correlated with the signal: when the signal fluctuates, the shot term in
    the denominator moves with it and half-cancels the change. These
    contributions sum to ``"combined"``.

    Returns one entry per :data:`NOISE_SOURCES` (each source's contribution to
    the bar) plus ``"combined"`` (the full error bar = their sum). Multiply any
    entry by the configured ``sigma`` to get the plotted half-width (in SNR
    units, added to / subtracted from the nominal SNR).
    """
    keys = (*NOISE_SOURCES, "combined")
    if noise.total <= 0:
        return {k: 0.0 for k in keys}
    inv_total_var = 1.0 / noise.total ** 2
    out = {}
    for s in NOISE_SOURCES:
        var_frac = getattr(noise, s) ** 2 * inv_total_var
        out[s] = 0.5 * var_frac if s == "target_shot" else var_frac
    out["combined"] = sum(out[s] for s in NOISE_SOURCES)
    return out
