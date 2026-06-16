"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


def data_dir() -> Path:
    """Return the project data directory."""
    return Path(__file__).parent.parent / "data"


def resolve_data_path(relative: str) -> Path:
    """Resolve a path relative to the package data directory."""
    return data_dir() / relative


# ── Config models ──────────────────────────────────────────────────────────


class ThermalConfig(BaseModel):
    enabled: bool = False
    temperature: float = 300.0   # K
    emissivity: float = 0.9
    absorptivity: float = 0.9


class TargetConfig(BaseModel):
    albedo: float = 0.2
    range: float = 2.0e9         # meters
    phase_angles: list[float] = [85.0]
    thermal: ThermalConfig = ThermalConfig()


class AtmosphereConfig(BaseModel):
    enabled: bool = True
    transmission: float = 0.7
    sky_bkg_mv: float = 20.0     # mag/arcsec^2
    r0: float = 0.11             # Fried parameter, meters
    r0_wavelength: float = 0.65e-6  # meters
    # Scintillation (Young 1967 / Osborn 2015). ``scintillation_coeff`` is the
    # empirical Young coefficient C_Y (theoretical ~1.5); 0 disables the term.
    # It adds a signal-proportional noise -> a magnitude-independent SNR ceiling
    # of 1/sigma that scales as sqrt(t). Only applied when ``enabled`` (no
    # scintillation in space). Fit ``scintillation_coeff`` to the bright-star
    # SNR plateau for a given site; ``airmass`` defaults to 1 to match
    # airmass-normalized (zenith) on-sky data.
    scintillation_coeff: float = 0.0     # Young C_Y; 0 = off
    airmass: float = 1.0                 # sec(z); scintillation noise ∝ X^3
    observatory_altitude: float = 0.0    # meters; scintillation ∝ exp(-2h/H)


class OpticsConfig(BaseModel):
    name: str = "Telescope"
    aperture_diameter: float = 0.508   # meters
    focal_length: float = 3.454        # meters
    obscuration: float = 0.15          # fraction by area
    image_circle: float | None = None  # mm
    jitter: float = 1.0                # arcsec RMS
    step_settle_time: float = 5.0      # seconds, slew + settle between fields
    throughput_files: list[str] = []


class FilterConfig(BaseModel):
    name: str | None = None
    file: str | None = None

    @model_validator(mode='before')
    @classmethod
    def _normalize_none_strings(cls, data):
        """Allow 'none' as a synonym for null in YAML."""
        if isinstance(data, dict):
            for key in ('name', 'file'):
                if isinstance(data.get(key), str) and data[key].lower() == 'none':
                    data[key] = None
        return data


class DetectorConfig(BaseModel):
    name: str = "Detector"
    pixel_size: float = 10.0e-6   # meters
    height: int = 4096            # pixels
    width: int = 4096             # pixels
    dark_current: float = 0.15    # e-/pix/s
    read_noise: float = 0.8       # e-/pix
    full_well: float = 80000.0    # e-
    bit_depth: int = 16
    is_cmos: bool = True
    frame_rate: float | None = None  # Hz
    qe_file: str = "detectors/qe-imx455.csv"
    # Signal-proportional systematic floor (flat-field / PSF-model residual), as
    # a fraction of source signal. Like scintillation it is multiplicative, but
    # exposure-independent, so it sets the asymptotic bright-end SNR ceiling of
    # 1/systematic_floor that even long exposures cannot beat. 0 = off.
    systematic_floor: float = 0.0


class ObservationConfig(BaseModel):
    exposure_time: float = 1.0         # seconds (fixed point for SNR-vs-mV)
    exposure_range: list[float] = [0.1, 10.0]
    mv_range: list[float] = [14.0, 21.0]
    mv_step: float = 0.5
    binning: int = 1
    num_apertures: int = 1
    num_fields: int = 1
    num_frames: int = 1
    snr_threshold: float = 6.0
    streak_rate: float = 0.0          # pix/s (0 = rate-track mode)
    # Longest exposure the search-rate root-find may integrate to. None auto-
    # selects the exposure that just reaches the faintest magnitude, so the rate
    # tapers smoothly to ~0 at the faint edge; set a number (seconds) to cap it
    # at a realistic value (the faint tail then drops to 0 once it's unreachable).
    max_search_exposure_s: float | None = None


class OutputConfig(BaseModel):
    plots: list[str] = ["snr_vs_t", "snr_vs_mv", "search_rate"]
    log_level: str = "INFO"
    # SNR error-band settings. ``sigma`` is the band half-width in standard
    # deviations (set to 0 to disable the band). By default a single combined
    # band is drawn -- the SNR uncertainty from every noise source except the
    # target's own (irreducible) shot noise -- so it responds to read noise,
    # dark current, sky brightness, bit depth, etc. Set ``visualize_noise:
    # true`` to additionally overplot each individual noise source's band.
    sigma: float = Field(default=1.0, ge=0.0)
    visualize_noise: bool = False


class SNRConfig(BaseModel):
    target: TargetConfig = TargetConfig()
    atmosphere: AtmosphereConfig = AtmosphereConfig()
    optics: OpticsConfig = OpticsConfig()
    filter: FilterConfig = FilterConfig()
    detector: DetectorConfig = DetectorConfig()
    observation: ObservationConfig = ObservationConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path) -> SNRConfig:
    """Load configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SNRConfig(**(raw or {}))
