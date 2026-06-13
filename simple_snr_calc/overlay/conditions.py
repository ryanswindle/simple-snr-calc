"""Per-night measured observing conditions and how they map onto model config.

senpai writes a ``nights_summary.csv`` aggregating each night's measured
conditions (extinction, transmission, sky brightness, seeing, limiting mag).
We translate those measurements into overrides on an :class:`SNRConfig` so the
model is evaluated under the same sky the on-sky data was taken in -- making a
model-vs-data overlay a fair comparison rather than design-spec vs reality.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from ..config import SNRConfig
from ..optics import compute_ifov

# senpai nights_summary.csv headers -> NightConditions fields.
_SUMMARY_KEYS = {
    "night": "night_id",
    "moon%": "moon_illumination",
    "moonSep°": "moon_sep_deg",
    "k": "extinction_k",
    "T_zen": "zenith_transmission",
    "FWHM_px": "fwhm_px",
    "sky_μ": "sky_mag_arcsec2",
    "lim50": "limiting_mag_50",
}


@dataclass
class NightConditions:
    """Measured conditions for one night, as read from senpai's summary.

    All fields are optional: a missing/blank cell becomes ``None`` and the
    corresponding model knob is left at its config default by
    :func:`apply_conditions`.
    """

    night_id: str
    extinction_k: float | None = None        # mag / airmass
    zenith_transmission: float | None = None  # T_zen, fraction (0..1)
    sky_mag_arcsec2: float | None = None      # sky brightness, mag/arcsec^2
    fwhm_px: float | None = None              # median stellar FWHM, native pixels
    limiting_mag_50: float | None = None      # 50%-completeness limiting mag
    moon_illumination: float | None = None    # fraction (0..1)
    moon_sep_deg: float | None = None

    @classmethod
    def from_summary_row(cls, row: dict[str, str]) -> "NightConditions":
        def _f(key: str) -> float | None:
            v = row.get(key)
            if v is None or str(v).strip() in ("", "—", "nan", "None"):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        return cls(
            night_id=str(row.get("night", "")).strip(),
            extinction_k=_f("k"),
            zenith_transmission=_f("T_zen"),
            sky_mag_arcsec2=_f("sky_μ"),
            fwhm_px=_f("FWHM_px"),
            limiting_mag_50=_f("lim50"),
            moon_illumination=_f("moon%"),
            moon_sep_deg=_f("moonSep°"),
        )


def load_nights_summary(csv_path: str | Path) -> dict[str, NightConditions]:
    """Load senpai's ``nights_summary.csv`` into ``{night_id: NightConditions}``."""
    out: dict[str, NightConditions] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cond = NightConditions.from_summary_row(row)
            if cond.night_id:
                out[cond.night_id] = cond
    return out


def r0_from_fwhm(fwhm_arcsec: float, wavelength_m: float) -> float:
    """Invert the Kolmogorov seeing relation ``FWHM = 0.98 * lambda / r0``.

    Returns the Fried parameter (m) that reproduces ``fwhm_arcsec`` at
    ``wavelength_m`` under simple-snr-calc's seeing model (see
    :func:`simple_snr_calc.atmosphere.seeing_sigma_rad`).
    """
    fwhm_rad = math.radians(fwhm_arcsec / 3600.0)
    if fwhm_rad <= 0:
        raise ValueError(f"non-positive FWHM: {fwhm_arcsec} arcsec")
    return 0.98 * wavelength_m / fwhm_rad


def fwhm_px_to_arcsec(fwhm_px: float, config: SNRConfig) -> float:
    """Convert a FWHM in native detector pixels to arcsec for this instrument."""
    ifov_arcsec = compute_ifov(config.detector.pixel_size, config.optics.focal_length)
    return fwhm_px * ifov_arcsec


def apply_conditions(
    config: SNRConfig,
    cond: NightConditions,
    *,
    overhead_s: float | None = None,
    target_snr: float | None = None,
    mv_range: tuple[float, float] | None = None,
    fold_fwhm_into_seeing: bool = True,
) -> SNRConfig:
    """Return a deep copy of ``config`` with this night's measurements applied.

    Mapping
    -------
    * ``zenith_transmission`` -> ``atmosphere.transmission`` (the model applies
      this as a flat multiplier on signal; the on-sky SNRs are airmass-normalized
      to airmass=1, i.e. the zenith, so T_zen is the matching transmission).
    * ``sky_mag_arcsec2``     -> ``atmosphere.sky_bkg_mv``.
    * ``fwhm_px``             -> ``atmosphere.r0`` via :func:`r0_from_fwhm`. The
      measured on-sky FWHM already folds in seeing + jitter + optics, so when
      ``fold_fwhm_into_seeing`` is set we zero ``optics.jitter`` to avoid adding
      it a second time -- the seeing term alone is solved to reproduce the
      measured total FWHM.

    Extras (not from the CSV; supplied by the caller from senpai's plot_data):
    * ``overhead_s``  -> ``optics.step_settle_time`` so the model's per-field
      duty cycle matches senpai's fitted grid cadence.
    * ``target_snr``  -> ``observation.snr_threshold`` so the model's search-rate
      detection threshold matches senpai's target sigma.
    * ``mv_range``    -> ``observation.mv_range`` to cover the on-sky data extent.
    """
    cfg = config.model_copy(deep=True)

    if cond.zenith_transmission is not None:
        cfg.atmosphere.transmission = cond.zenith_transmission
    if cond.sky_mag_arcsec2 is not None:
        cfg.atmosphere.sky_bkg_mv = cond.sky_mag_arcsec2
    if cond.fwhm_px is not None:
        fwhm_arcsec = fwhm_px_to_arcsec(cond.fwhm_px, cfg)
        cfg.atmosphere.r0 = r0_from_fwhm(fwhm_arcsec, cfg.atmosphere.r0_wavelength)
        if fold_fwhm_into_seeing:
            cfg.optics.jitter = 0.0

    if overhead_s is not None:
        cfg.optics.step_settle_time = overhead_s
    if target_snr is not None:
        cfg.observation.snr_threshold = target_snr
    if mv_range is not None:
        cfg.observation.mv_range = [float(mv_range[0]), float(mv_range[1])]

    return cfg
