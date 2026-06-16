"""Overlay simple-snr-calc model curves on senpai on-sky calibration plots.

This subpackage couples the SNR model to `senpai`'s nightly calibration outputs
(a night's ``calibration/`` folder: ``plot_data.json`` for the on-sky panels and
``night_calibration.json`` for that night's measured conditions) so a model curve
can be drawn on the same axes as the measured on-sky data, evaluated under the
*measured* sky conditions for that night (transmission, sky brightness, seeing).
Everything is read from the night's own ``calibration/`` folder; no cross-night
aggregate (e.g. nights_summary.csv) is used.

senpai is an optional dependency of simple-snr-calc -- install it with the
``overlay`` extra (``pip install simple-snr-calc[overlay]``) or have
``astro-senpai`` importable. The model side has no senpai dependency; only the
plot-data loaders here read senpai's JSON schema (which is plain JSON, so senpai
need not be importable to *load* it -- only to *regenerate* it from raw batches).
"""

from .conditions import (
    NightConditions,
    load_night_conditions,
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
    "load_night_conditions",
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
