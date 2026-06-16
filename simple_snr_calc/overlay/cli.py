"""CLI for overlaying model curves on senpai calibration plots.

    snr-overlay search-rate <run_dir> [--config YAML] [--out PNG]
    snr-overlay all         <run_dir> [--config YAML] [--out PREFIX]

``run_dir`` is a senpai night run directory containing a ``calibration/`` folder
with ``plot_data.json`` and ``night_calibration.json`` (e.g. ``.../DAO01_20260602``).
Everything the overlay needs -- the on-sky panels and that night's measured sky
conditions -- comes from ``calibration/``; no cross-night aggregate is read.

Every overlay is written twice: the titled PNG at ``--out`` plus a title-less
``*_clean.png`` twin for paper figures. The ``all`` command runs every overlay
type, treating ``--out`` as a filename prefix.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..config import load_config
from .search_rate import plot_search_rate_overlay
from .snr_vs_mag import plot_snr_vs_mag_overlay
from .snr_vs_exposure import plot_snr_vs_exposure_overlay


def _calib_inputs(run_dir: Path):
    """Return ``(plot_data.json, night_calibration.json)`` under ``run_dir``."""
    calib = run_dir / "calibration"
    plot_data = calib / "plot_data.json"
    night_cal = calib / "night_calibration.json"
    if not plot_data.exists():
        raise click.ClickException(f"no plot_data.json at {plot_data}")
    if not night_cal.exists():
        raise click.ClickException(f"no night_calibration.json at {night_cal}")
    return plot_data, night_cal


def _resolve_paths(run_dir: Path, out: Path | None, default_name: str):
    plot_data, night_cal = _calib_inputs(run_dir)
    if out is None:
        out = run_dir / "calibration" / default_name
    return plot_data, night_cal, Path(out)


_DAO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dao.yaml"


@click.group()
def main():
    """Overlay simple-snr-calc model curves on senpai calibration plots."""


@main.command("search-rate")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG (default: <run_dir>/calibration/search_rate_overlay.png).")
@click.option("--mv-step", type=float, default=0.1, show_default=True,
              help="Magnitude step for the model curve.")
@click.option("--max-exposure", type=float, default=None,
              help="Max model exposure (s). Default: auto-fit to the faint edge. "
                   "Pass e.g. 10 to cap at the on-sky frame exposure.")
def search_rate(run_dir: Path, config_path, out, mv_step, max_exposure):
    """Search-rate overlay for one night run directory."""
    plot_data, night_cal, out = _resolve_paths(
        run_dir, out, "search_rate_overlay.png")
    config = load_config(config_path or _DAO_CONFIG)
    for p in plot_search_rate_overlay(
        plot_data, night_cal, config, out,
        mv_step=mv_step, max_exposure_s=max_exposure,
    ):
        click.echo(f"wrote {p}")


@main.command("snr-vs-mag")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG (default: <run_dir>/calibration/snr_vs_mag_overlay.png).")
@click.option("--mv-step", type=float, default=0.1, show_default=True,
              help="Magnitude step for the model curves.")
def snr_vs_mag(run_dir: Path, config_path, out, mv_step):
    """SNR-vs-magnitude overlay for one night run directory."""
    plot_data, night_cal, out = _resolve_paths(
        run_dir, out, "snr_vs_mag_overlay.png")
    config = load_config(config_path or _DAO_CONFIG)
    for p in plot_snr_vs_mag_overlay(
            plot_data, night_cal, config, out, mv_step=mv_step):
        click.echo(f"wrote {p}")


@main.command("snr-vs-exposure")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG (default: <run_dir>/calibration/snr_vs_exposure_overlay.png).")
def snr_vs_exposure(run_dir: Path, config_path, out):
    """SNR-vs-exposure (by magnitude) overlay for one night run directory."""
    plot_data, night_cal, out = _resolve_paths(
        run_dir, out, "snr_vs_exposure_overlay.png")
    config = load_config(config_path or _DAO_CONFIG)
    for p in plot_snr_vs_exposure_overlay(plot_data, night_cal, config, out):
        click.echo(f"wrote {p}")


@main.command("all")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Base model config YAML (default: configs/dao.yaml).")
@click.option("--out", "out_prefix", type=click.Path(path_type=Path), default=None,
              help="Output filename prefix. Each overlay is written as "
                   "<out>_<type>.png plus a title-less <out>_<type>_clean.png. "
                   "Default: <run_dir>/calibration/<run_dir name>.")
@click.option("--mv-step", type=float, default=0.1, show_default=True,
              help="Magnitude step for the model curves.")
@click.option("--max-exposure", type=float, default=None,
              help="Max model exposure (s) for the search-rate overlay. "
                   "Default: auto-fit to the faint edge.")
def all_overlays(run_dir: Path, config_path, out_prefix, mv_step, max_exposure):
    """Generate all overlays (each titled + a _clean twin) for one night run dir."""
    plot_data, night_cal = _calib_inputs(run_dir)
    config = load_config(config_path or _DAO_CONFIG)
    prefix = (Path(out_prefix) if out_prefix is not None
              else run_dir / "calibration" / run_dir.name)

    jobs = [
        ("search_rate", lambda o: plot_search_rate_overlay(
            plot_data, night_cal, config, o,
            mv_step=mv_step, max_exposure_s=max_exposure)),
        ("snr_vs_mag", lambda o: plot_snr_vs_mag_overlay(
            plot_data, night_cal, config, o, mv_step=mv_step)),
        ("snr_vs_exposure", lambda o: plot_snr_vs_exposure_overlay(
            plot_data, night_cal, config, o)),
    ]
    for kind, render in jobs:
        try:
            for p in render(f"{prefix}_{kind}.png"):
                click.echo(f"wrote {p}")
        except (ValueError, KeyError) as e:
            # A night may legitimately lack a given on-sky panel; skip, don't abort.
            click.echo(f"skipped {kind}: {e}", err=True)


if __name__ == "__main__":
    main()
