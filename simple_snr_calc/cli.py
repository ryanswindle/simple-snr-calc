"""Command-line interface."""

import sys

import click
import matplotlib.pyplot as plt
from loguru import logger


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option(
    "--plot", "-p", multiple=True,
    type=click.Choice(["snr_vs_t", "snr_vs_mv", "search_rate", "all"]),
    default=["all"],
    help="Plots to display (default: all)",
)
def main(config_file, debug, plot):
    """Run SNR calculation from a YAML config file."""
    from .config import load_config
    from .snr import SNRCalculator
    from .plotting import plot_summary

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    logger.info(f"Loading config from {config_file}")
    config = load_config(config_file)

    logger.info("Running SNR sweep...")
    calc = SNRCalculator(config)
    results = calc.sweep()

    logger.info("Generating plots...")
    fig = plot_summary(config, results)
    plt.show()
