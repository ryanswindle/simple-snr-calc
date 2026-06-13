"""Overlay simple-snr-calc model curves on senpai on-sky calibration plots.

This subpackage couples the SNR model to `senpai`'s nightly calibration outputs
(``plot_data.json`` + ``nights_summary.csv``) so a model curve can be drawn on
the same axes as the measured on-sky data, evaluated under the *measured* sky
conditions for that night (transmission, sky brightness, seeing).

senpai is an optional dependency of simple-snr-calc -- install it with the
``overlay`` extra (``pip install simple-snr-calc[overlay]``) or have
``astro-senpai`` importable. The model side has no senpai dependency; only the
plot-data loaders here read senpai's JSON schema (which is plain JSON, so senpai
need not be importable to *load* it -- only to *regenerate* it from raw batches).
"""

from .conditions import (
    NightConditions,
    load_nights_summary,
    apply_conditions,
    r0_from_fwhm,
)
from .search_rate import (
    load_plot_data,
    model_search_rate_vs_mag,
    plot_search_rate_overlay,
)
from .snr_vs_mag import (
    model_snr_vs_mag,
    plot_snr_vs_mag_overlay,
)
from .snr_vs_exposure import (
    model_snr_vs_exposure,
    plot_snr_vs_exposure_overlay,
)

__all__ = [
    "NightConditions",
    "load_nights_summary",
    "apply_conditions",
    "r0_from_fwhm",
    "load_plot_data",
    "model_search_rate_vs_mag",
    "plot_search_rate_overlay",
    "model_snr_vs_mag",
    "plot_snr_vs_mag_overlay",
    "model_snr_vs_exposure",
    "plot_snr_vs_exposure_overlay",
]
