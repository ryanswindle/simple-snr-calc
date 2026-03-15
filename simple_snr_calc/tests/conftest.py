"""Shared fixtures for tests."""

import sys

import pytest
from loguru import logger

from simple_snr_calc.config import (
    SNRConfig,
    TargetConfig,
    AtmosphereConfig,
    OpticsConfig,
    FilterConfig,
    DetectorConfig,
    ObservationConfig,
    ThermalConfig,
)
from simple_snr_calc.snr import SNRCalculator


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress loguru output during tests."""
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    yield


@pytest.fixture
def ground_config():
    """A baseline ground-based configuration."""
    return SNRConfig(
        target=TargetConfig(
            albedo=0.2,
            range=2e9,
            phase_angles=[85.0],
            thermal=ThermalConfig(enabled=False),
        ),
        atmosphere=AtmosphereConfig(
            enabled=True,
            transmission=0.7,
            sky_bkg_mv=20.0,
            r0=0.11,
            r0_wavelength=0.65e-6,
        ),
        optics=OpticsConfig(
            name="Test Telescope",
            aperture_diameter=0.508,
            focal_length=3.454,
            obscuration=0.15,
            image_circle=52.0,
            jitter=1.0,
            throughput_files=[
                "optics/mirror-al.csv",
                "optics/mirror-al.csv",
            ],
        ),
        filter=FilterConfig(name=None),
        detector=DetectorConfig(
            name="Test Detector",
            pixel_size=10.0e-6,
            height=8120,
            width=8120,
            dark_current=0.15,
            read_noise=0.8,
            full_well=80000,
            bit_depth=16,
            is_cmos=True,
            frame_rate=2.9,
            qe_file="detectors/qe-teledyne-cosmos-8k.csv",
        ),
        observation=ObservationConfig(
            exposure_time=1.0,
            exposure_range=[0.1, 10.0],
            mv_range=[14.0, 21.0],
            mv_step=1.0,
            binning=1,
            num_apertures=1,
            num_fields=1,
            num_frames=1,
            snr_threshold=6.0,
            streak_rate=0.0,
        ),
    )


@pytest.fixture
def calculator(ground_config):
    """An SNRCalculator from the baseline ground config."""
    return SNRCalculator(ground_config)
