# simple-snr-calc

![CI](https://github.com/ryanswindle/simple-snr-calc/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/ryanswindle/simple-snr-calc/graph/badge.svg?token=M58Y2KTR2V)](https://codecov.io/gh/ryanswindle/simple-snr-calc)

Signal-to-noise ratio calculator for remote sensing of satellites. Supports ground-based and space-based sensors across UV, optical, near-IR, and IR wavelengths.

## Features

- **Reflected solar + thermal emission**: Lambertian sphere model with Tousey/Sussman phase angle factors, plus Planck blackbody thermal emission (optional)
- **Ground- and space-based sensors**: Atmosphere (extinction, sky background, seeing) can be enabled or disabled
- **Full throughput chain**: Optics coatings, spectral filters, and detector QE curves multiplied on a common wavelength grid
- **Detector noise model**: Shot noise, sky background, dark current, read noise, quantization noise with proper CCD/CMOS binning
- **Search rate**: Computes sky survey rate at a given SNR threshold
- **Peak pixel fraction**: Analytical ensquared energy and blur model for streaked targets
- **YAML configuration**: Single config file defines the entire observation scenario
- **CLI and Python API**: Run from the command line or import into your own code

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd simple-snr-calc

# Create a virtual environment and install
uv venv
uv pip install -e "."
```

## Quick Start

### CLI

```bash
# Run with an example config
simple-snr-calc configs/example_ground.yaml

# Debug logging
simple-snr-calc configs/example_ground.yaml --debug
```

### Python API

```python
from simple_snr_calc import load_config, SNRCalculator

# Load a YAML config
config = load_config("configs/example_ground.yaml")
calc = SNRCalculator(config)

# Single-point SNR
result = calc.compute_snr(mv=16.0, exposure_time=1.0)
print(f"SNR = {result.snr:.2f}")
print(f"Noise breakdown: {result.noise.variance_fractions()}")

# Full sweep (mV range x exposure time range)
results = calc.sweep()
```

### Programmatic Config

```python
from simple_snr_calc import SNRConfig, SNRCalculator
from simple_snr_calc.config import (
    TargetConfig, AtmosphereConfig, OpticsConfig,
    DetectorConfig, ObservationConfig,
)

config = SNRConfig(
    target=TargetConfig(albedo=0.2, range=2e9),
    atmosphere=AtmosphereConfig(enabled=False),  # space-based
    optics=OpticsConfig(
        aperture_diameter=0.3,
        focal_length=1.5,
        throughput_files=["optics/mirror-pwi-al.csv"],
    ),
    detector=DetectorConfig(qe_file="detectors/qe-imx455.csv"),
    observation=ObservationConfig(mv_range=[10.0, 18.0]),
)
calc = SNRCalculator(config)
```

### Utilities

Individual functions are importable for use in other tools:

```python
from simple_snr_calc import (
    load_solar_radiance,        # solar spectrum data
    planck_spectral_radiance,   # Planck blackbody B(lambda, T)
    load_curve,                 # load any CSV curve (QE, filter, etc.)
    phase_angle_factor,         # Lambertian sphere phase function
    radius_from_mv,             # apparent magnitude -> target radius
    build_throughput,           # optics x filter x QE chain
    compute_ensquared_energy,   # analytical Gaussian ensquared energy
    compute_search_rates,       # survey search rate
)
```

## Configuration

All parameters are set in a single YAML file. See `configs/example_ground.yaml` and `configs/example_space.yaml` for annotated examples.

Key sections:

| Section | Description |
|---|---|
| `target` | Albedo, range, phase angles, thermal emission settings |
| `atmosphere` | Enable/disable, transmission, sky background, Fried parameter |
| `optics` | Aperture, focal length, obscuration, jitter, throughput curves |
| `filter` | Filter name (looked up in `data/filters/`) or explicit file path |
| `detector` | Pixel size, dimensions, noise parameters, QE curve |
| `observation` | Exposure time, magnitude range, binning, coadding, SNR threshold |

Set `atmosphere.enabled: false` for space-based sensors. Set `target.thermal.enabled: true` to include thermal emission.

## Data Files

Spectral data lives in `data/` with the following structure:

```
data/
  solar/       Solar radiance spectra
  optics/      Mirror and lens throughput curves
  filters/     Spectral filter transmission curves (Johnson-Cousins, Bayer, etc.)
  detectors/   Detector QE curves
  psf/         Blur efficiency lookup table
```

Curves are 2-column CSV files (wavelength in nm, fractional value) and can be added by placing new files in the appropriate directory.

## License

See [LICENSE](LICENSE).
