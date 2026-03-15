"""Core SNR calculator."""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from .config import SNRConfig
from .spectrum import load_solar_radiance, watts_to_photons
from .target import (
    solar_irradiance_at_target,
    radius_from_mv,
    thermal_photon_rate,
)
from .atmosphere import sky_background_power, combined_psf_sigma_rad
from .optics import (
    build_throughput,
    compute_ifov,
    compute_fov,
    compute_ensquared_energy,
    peak_pixel_fraction,
    compute_zero_point,
    watts_to_photons_factor,
)
from .noise import compute_noise, NoiseBudget
from .search_rate import compute_search_rates


@dataclass
class SNRResult:
    """Result of a single SNR calculation."""
    snr: float
    saturated: bool
    peak_signal: float
    noise: NoiseBudget
    thermal_photons_s: float = 0.0


@dataclass
class SweepResult:
    """Result of a full magnitude/exposure-time sweep."""
    mvs: np.ndarray
    exposure_times: np.ndarray
    snr_grid: np.ndarray            # shape (n_mv, n_t)
    saturated_grid: np.ndarray      # shape (n_mv, n_t), bool
    snr_at_fixed_t: np.ndarray      # shape (n_mv,)
    saturated_at_fixed_t: np.ndarray
    search_rates: np.ndarray        # shape (n_mv,)
    # Derived quantities
    ifov: float
    fov: tuple[float, float]
    fwhm_arcsec: float
    ensquared_energy: float
    zero_point: float
    throughput: np.ndarray = field(default=None, repr=False)
    wl_nm: np.ndarray = field(default=None, repr=False)


class SNRCalculator:
    """Stateful SNR calculator that precomputes shared quantities.

    Usage
    -----
    >>> from simple_snr_calc import load_config, SNRCalculator
    >>> config = load_config("configs/example_ground.yaml")
    >>> calc = SNRCalculator(config)
    >>> result = calc.compute_snr(mv=16.0, exposure_time=1.0)
    >>> print(result.snr)
    """

    def __init__(self, config: SNRConfig):
        self.config = config
        self._setup()

    def _setup(self):
        """Precompute quantities that don't change with mV or exposure time."""
        cfg = self.config

        # Load solar spectrum
        self.wl_nm, solar_rad_W = load_solar_radiance()
        solar_rad_ph = watts_to_photons(self.wl_nm, solar_rad_W)

        # Solar irradiance at target (1 AU)
        self.solar_irr_W = solar_irradiance_at_target(solar_rad_W)
        self.solar_irr_ph = solar_irradiance_at_target(solar_rad_ph)

        # Build throughput
        self.throughput = build_throughput(
            self.wl_nm,
            cfg.optics.throughput_files,
            cfg.filter.name,
            cfg.filter.file,
            cfg.detector.qe_file,
        )

        # Effective aperture area
        self.aperture_area = (
            np.pi * (cfg.optics.aperture_diameter / 2.0) ** 2
            * (1.0 - cfg.optics.obscuration)
        )

        # Zero point
        self.zero_point = compute_zero_point(
            self.wl_nm, self.solar_irr_ph, self.throughput, self.aperture_area
        )
        logger.debug(f"Zero point: {self.zero_point:.2f} mag")

        # IFOV
        self.ifov = compute_ifov(cfg.detector.pixel_size, cfg.optics.focal_length)
        logger.debug(f"IFOV: {self.ifov:.3f} arcsec")

        # FOV
        self.fov = compute_fov(
            self.ifov,
            cfg.detector.width,
            cfg.detector.height,
            cfg.optics.image_circle,
            cfg.detector.pixel_size,
        )
        logger.debug(f"FOV: {self.fov[0]:.3f} x {self.fov[1]:.3f} deg")

        # PSF and ensquared energy
        if cfg.atmosphere.enabled:
            self.psf_sigma = combined_psf_sigma_rad(
                cfg.atmosphere.r0,
                cfg.atmosphere.r0_wavelength,
                cfg.optics.jitter,
            )
        else:
            # Space-based: jitter only
            self.psf_sigma = cfg.optics.jitter * 4.848e-6

        self.ensquared_energy, _ = compute_ensquared_energy(
            self.psf_sigma, self.ifov, cfg.observation.binning
        )

        fwhm_rad = self.psf_sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))
        self.fwhm_arcsec = fwhm_rad * 206265.0
        logger.debug(
            f"PSF FWHM: {self.fwhm_arcsec:.2f} arcsec, "
            f"EE: {self.ensquared_energy:.4f}"
        )

        # Watts-to-photons conversion factor (for sky background)
        self.eta = watts_to_photons_factor(
            self.wl_nm, self.solar_irr_W, self.throughput
        )

        # Sky background rate (e-/s/pixel)
        if cfg.atmosphere.enabled:
            sky_power = sky_background_power(
                cfg.atmosphere.sky_bkg_mv, cfg.optics.aperture_diameter
            )
            binned_ifov_arcsec = self.ifov * cfg.observation.binning
            self.sky_bkg_rate = (
                sky_power
                * binned_ifov_arcsec ** 2
                * self.eta
                * (1.0 - cfg.optics.obscuration)
            )
        else:
            self.sky_bkg_rate = 0.0

        logger.debug(f"Sky background rate: {self.sky_bkg_rate:.2f} e-/s/pixel")

    def compute_snr(self, mv: float, exposure_time: float) -> SNRResult:
        """Compute SNR for a given apparent magnitude and exposure time."""
        cfg = self.config

        # Reflected photons/s at FPA
        reflected_pps = 10.0 ** ((self.zero_point - mv) / 2.5)

        # Thermal photons/s at FPA
        thermal_pps = 0.0
        if cfg.target.thermal.enabled:
            r_target = radius_from_mv(
                mv, cfg.target.albedo, cfg.target.range,
                cfg.target.phase_angles[0],
            )
            thermal_pps = thermal_photon_rate(
                self.wl_nm, self.throughput,
                cfg.target.thermal.temperature,
                cfg.target.thermal.emissivity,
                r_target, cfg.target.range,
                self.aperture_area,
            )

        # Apply atmospheric transmission
        atm_trans = cfg.atmosphere.transmission if cfg.atmosphere.enabled else 1.0
        total_pps = (reflected_pps + thermal_pps) * atm_trans

        # Peak pixel fraction
        streak_length = cfg.observation.streak_rate * exposure_time
        peak_frac = peak_pixel_fraction(self.ensquared_energy, streak_length)
        peak_signal = total_pps * exposure_time * peak_frac

        # Noise budget
        noise = compute_noise(
            peak_signal=peak_signal,
            sky_bkg_rate=self.sky_bkg_rate,
            dark_current=cfg.detector.dark_current,
            read_noise=cfg.detector.read_noise,
            full_well=cfg.detector.full_well,
            bit_depth=cfg.detector.bit_depth,
            exposure_time=exposure_time,
            binning=cfg.observation.binning,
            is_cmos=cfg.detector.is_cmos,
        )

        # SNR (single frame)
        snr = peak_signal / noise.total if noise.total > 0 else 0.0

        # Coadding
        num_coadds = cfg.observation.num_apertures * cfg.observation.num_frames
        snr *= np.sqrt(num_coadds)

        # Saturation check
        total_e = (
            peak_signal
            + (self.sky_bkg_rate + cfg.detector.dark_current) * exposure_time
            + cfg.detector.read_noise
        )
        saturated = total_e >= cfg.detector.full_well

        logger.debug(
            f"mV={mv:.1f}, t={exposure_time:.3f}s: "
            f"signal={peak_signal:.1f}e-, SNR={snr:.2f}, "
            f"thermal={thermal_pps:.1f}ph/s"
        )

        return SNRResult(
            snr=snr,
            saturated=saturated,
            peak_signal=peak_signal,
            noise=noise,
            thermal_photons_s=thermal_pps,
        )

    def sweep(self) -> SweepResult:
        """Run SNR calculation over full mV and exposure time ranges."""
        cfg = self.config
        obs = cfg.observation

        mvs = np.arange(
            obs.mv_range[0], obs.mv_range[1] + obs.mv_step / 2, obs.mv_step
        )
        its = np.linspace(obs.exposure_range[0], obs.exposure_range[1], 49)

        snr_grid = np.zeros((len(mvs), len(its)))
        sat_grid = np.zeros((len(mvs), len(its)), dtype=bool)
        snr_at_fixed_t = np.zeros(len(mvs))
        sat_at_fixed_t = np.zeros(len(mvs), dtype=bool)

        for i, mv in enumerate(mvs):
            # Fixed exposure time
            result = self.compute_snr(mv, obs.exposure_time)
            snr_at_fixed_t[i] = result.snr
            sat_at_fixed_t[i] = result.saturated

            # Sweep exposure times
            for j, t in enumerate(its):
                result = self.compute_snr(mv, t)
                snr_grid[i, j] = result.snr
                sat_grid[i, j] = result.saturated

            logger.info(
                f"mV={mv:.1f}: SNR@t={obs.exposure_time}s = "
                f"{snr_at_fixed_t[i]:.1f}"
            )

        # Search rates
        search_rates = compute_search_rates(
            mvs, snr_grid, its, self.fov, obs, cfg.detector.frame_rate,
        )

        return SweepResult(
            mvs=mvs,
            exposure_times=its,
            snr_grid=snr_grid,
            saturated_grid=sat_grid,
            snr_at_fixed_t=snr_at_fixed_t,
            saturated_at_fixed_t=sat_at_fixed_t,
            search_rates=search_rates,
            ifov=self.ifov,
            fov=self.fov,
            fwhm_arcsec=self.fwhm_arcsec,
            ensquared_energy=self.ensquared_energy,
            zero_point=self.zero_point,
            throughput=self.throughput,
            wl_nm=self.wl_nm,
        )
