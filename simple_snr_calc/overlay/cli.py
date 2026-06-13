"""CLI for overlaying model curves on senpai calibration plots.

    snr-overlay search-rate <run_dir> [--summary CSV] [--config YAML] [--out PNG]

``run_dir`` is a senpai night run directory containing
``calibration/plot_data.json`` (e.g. ``.../DAO01_20260602``).
"""

from __future__ import annotations

from pathlib import Path

import click

from ..config import load_config
from .search_rate import plot_search_rate_overlay
from .snr_vs_mag import plot_snr_vs_mag_overlay
from .snr_vs_exposure import plot_snr_vs_exposure_overlay


def _resolve_paths(run_dir: Path, summary: Path | None, out: Path | None,
                   default_name: str):
    plot_data = run_dir / "calibration" / "plot_data.json"
    if not plot_data.exists():
        raise click.ClickException(f"no plot_data.json at {plot_data}")
    if summary is None:
        # senpai writes nights_summary.csv at the root above the night dirs.
        summary = run_dir.parent / "nights_summary.csv"
    if not Path(summary).exists():
        raise click.ClickException(f"nights_summary.csv not found at {summary}")
    if out is None:
        out = run_dir / "calibration" / default_name
    return plot_data, Path(summary), Path(out)


_DAO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dao.yaml"


@click.group()
def main():
    """Overlay simple-snr-calc model curves on senpai calibration plots."""


@main.command("search-rate")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--summary", type=click.Path(path_type=Path), default=None,
              help="senpai nights_summary.csv (default: <run_dir>/../nights_summary.csv).")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG (default: <run_dir>/calibration/search_rate_overlay.png).")
@click.option("--mv-step", type=float, default=0.1, show_default=True,
              help="Magnitude step for the model curve.")
@click.option("--max-exposure", type=float, default=None,
              help="Max model exposure (s). Default: auto-fit to the faint edge. "
                   "Pass e.g. 10 to cap at the on-sky frame exposure.")
def search_rate(run_dir: Path, summary, config_path, out, mv_step, max_exposure):
    """Search-rate overlay for one night run directory."""
    plot_data, summary, out = _resolve_paths(
        run_dir, summary, out, "search_rate_overlay.png")
    config = load_config(config_path or _DAO_CONFIG)
    path = plot_search_rate_overlay(
        plot_data, summary, config, out,
        mv_step=mv_step, max_exposure_s=max_exposure,
    )
    click.echo(f"wrote {path}")


@main.command("snr-vs-mag")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--summary", type=click.Path(path_type=Path), default=None,
              help="senpai nights_summary.csv (default: <run_dir>/../nights_summary.csv).")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG (default: <run_dir>/calibration/snr_vs_mag_overlay.png).")
@click.option("--mv-step", type=float, default=0.1, show_default=True,
              help="Magnitude step for the model curves.")
def snr_vs_mag(run_dir: Path, summary, config_path, out, mv_step):
    """SNR-vs-magnitude overlay for one night run directory."""
    plot_data, summary, out = _resolve_paths(
        run_dir, summary, out, "snr_vs_mag_overlay.png")
    config = load_config(config_path or _DAO_CONFIG)
    path = plot_snr_vs_mag_overlay(
        plot_data, summary, config, out, mv_step=mv_step)
    click.echo(f"wrote {path}")


@main.command("snr-vs-exposure")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--summary", type=click.Path(path_type=Path), default=None,
              help="senpai nights_summary.csv (default: <run_dir>/../nights_summary.csv).")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG (default: <run_dir>/calibration/snr_vs_exposure_overlay.png).")
def snr_vs_exposure(run_dir: Path, summary, config_path, out):
    """SNR-vs-exposure (by magnitude) overlay for one night run directory."""
    plot_data, summary, out = _resolve_paths(
        run_dir, summary, out, "snr_vs_exposure_overlay.png")
    config = load_config(config_path or _DAO_CONFIG)
    path = plot_snr_vs_exposure_overlay(plot_data, summary, config, out)
    click.echo(f"wrote {path}")


if __name__ == "__main__":
    main()
