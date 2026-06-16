"""Noise budget computation."""

from dataclasses import dataclass
import numpy as np

# Additive variance components of the noise budget, in the order they are
# summed in quadrature to form ``NoiseBudget.total``. These are signal-
# *independent* in their fraction of the signal (shot grows as sqrt(S); the rest
# are fixed), so each one gets its own SNR error band when
# ``output.visualize_noise`` is set. See :data:`MULTIPLICATIVE_SOURCES` for the
# signal-proportional terms, which sit in ``total`` but carry no error band.
NOISE_SOURCES = (
    "target_shot",
    "sky_background",
    "dark_current",
    "read",
    "quantization",
)

# Multiplicative variance components: noise proportional to the source signal
# itself (sigma * S), so they impose a magnitude-independent SNR ceiling of
# 1/sigma rather than averaging down with brightness. They contribute to the
# total noise (and the variance budget) but *zero* to the SNR error band --
# being perfectly correlated with the signal, they cancel out of d(SNR)/d(S)
# (see :func:`snr_band_unit`).
MULTIPLICATIVE_SOURCES = (
    "scintillation",
    "systematic",
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
    scintillation: float = 0.0
    systematic: float = 0.0

    def variance_fractions(self) -> dict[str, float]:
        """Return noise variance fractions (noise power budget).

        Each value is the fraction of total noise variance contributed
        by that source. Fractions sum to 1.0 and cover every component --
        the additive :data:`NOISE_SOURCES` and the multiplicative
        :data:`MULTIPLICATIVE_SOURCES` (scintillation, systematic).
        """
        total_var = self.total ** 2
        if total_var == 0:
            return {}
        return {
            s: getattr(self, s) ** 2 / total_var
            for s in (*NOISE_SOURCES, *MULTIPLICATIVE_SOURCES)
        }


def compute_noise(peak_signal: float, sky_bkg_rate: float,
                  dark_current: float, read_noise: float,
                  full_well: float, bit_depth: int,
                  exposure_time: float, binning: int,
                  is_cmos: bool, scint_fraction: float = 0.0,
                  systematic_fraction: float = 0.0) -> NoiseBudget:
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
    scint_fraction : float
        Scintillation noise as a fraction of source signal (see
        :func:`simple_snr_calc.atmosphere.scintillation_fraction`). Adds
        ``scint_fraction * peak_signal`` electrons in quadrature.
    systematic_fraction : float
        Flat-field/PSF-model systematic floor as a fraction of source signal.
        Adds ``systematic_fraction * peak_signal`` electrons in quadrature.
    """
    signal = max(peak_signal, 0.0)
    target_shot = np.sqrt(signal)
    sky_shot = np.sqrt(max(sky_bkg_rate * exposure_time, 0.0))
    dark_shot = np.sqrt(max(dark_current * exposure_time, 0.0))

    # CMOS: digital binning sums N^2 pixels, each with independent read noise
    # CCD: on-chip binning, single readout
    rn = read_noise * binning if is_cmos else read_noise

    quant = full_well / 2 ** bit_depth / np.sqrt(12.0)

    # Multiplicative noise: proportional to the source signal itself. Applied to
    # the same (peak-pixel) signal the SNR uses, so it sets an SNR ceiling of
    # 1/sqrt(scint_fraction**2 + systematic_fraction**2) regardless of the
    # peak-vs-aperture convention (numerator and this term scale together).
    scint = max(scint_fraction, 0.0) * signal
    systematic = max(systematic_fraction, 0.0) * signal

    total = np.sqrt(
        target_shot ** 2
        + sky_shot ** 2
        + dark_shot ** 2
        + rn ** 2
        + quant ** 2
        + scint ** 2
        + systematic ** 2
    )

    return NoiseBudget(
        target_shot=target_shot,
        sky_background=sky_shot,
        dark_current=dark_shot,
        read=rn,
        quantization=quant,
        total=total,
        scintillation=scint,
        systematic=systematic,
    )


def snr_band_unit(noise: NoiseBudget) -> dict[str, float]:
    """SNR error-bar half-widths at sigma = 1, propagated through the SNR.

    Treating the measured signal as the random variable and letting the noise
    estimate track it (as real photometry does), the 1-sigma error bar on the
    *coadded* ``SNR = sqrt(Nc) * S / sqrt(S + B)`` works out to

        sigma(SNR) = (S/2 + B) / (S + B) = 1 - f_shot / 2 ,

    where ``S`` is the target-shot variance, ``B`` is every other *additive*
    variance (sky + dark + read + quantization), and ``f_shot =
    (target_shot/total)**2`` is the target-shot variance fraction. This is the
    honest bar: it is independent of the number of coadds (the sqrt(Nc) boost
    cancels between the SNR and the correspondingly sharpened flux estimate).

    The per-source *contributions* to the bar add linearly (each is
    proportional to that source's variance, not its RMS):

        target_shot:  f_shot / 2
        source X:     f_X = (sigma_X / total)**2     (sky, dark, read, quant)

    Target shot is down-weighted by 1/2 because it is the only *additive* source
    correlated with the signal: when the signal fluctuates, the shot term in
    the denominator moves with it and half-cancels the change. These
    contributions sum to ``"combined"``.

    Multiplicative noise (scintillation, systematic) is the limiting case of
    that correlation: it tracks the signal *exactly*, so ``SNR = S/(sigma*S) =
    1/sigma`` is insensitive to S and these terms drop out of d(SNR)/dS entirely
    -- their variance ``c*S**2`` cancels in the numerator, leaving the same
    ``(S/2 + B)`` over the now-larger ``total = S + B + c*S**2``. So they
    contribute *zero* to the bar yet shrink every other contribution. Hence,
    *without* multiplicative noise the bar runs from 1/2 (target-shot limited)
    to 1 (read/sky/dark/quant limited); *with* it dominating, "combined" falls
    below 1/2 toward 0 (a scintillation-limited SNR is essentially exact).

    Returns one entry per :data:`NOISE_SOURCES` (each source's contribution to
    the bar) plus ``"combined"`` (the full error bar = their sum); the
    :data:`MULTIPLICATIVE_SOURCES` have no entry (their contribution is zero by
    the cancellation above, but they still enter via ``total``). Multiply any
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
