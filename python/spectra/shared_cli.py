"""Shared helpers for spectroscopy command-line interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

from .output_names import default_output_path as default_named_output_path


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Show defaults while preserving example formatting."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help
        if not help_text:
            return ""
        if "%(default)" in help_text:
            return help_text
        if action.option_strings and action.default not in (None, False, argparse.SUPPRESS):
            return f"{help_text} (default: %(default)s)"
        return help_text


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    """Return the default NetCDF output path inside the project root."""
    return default_named_output_path(
        target_name="CO2",
        plot_type="spectrum",
        temperature_k=300.0,
        pressure_bar=1.0,
        wn_range=(20.0, 2500.0),
        suffix=".nc",
        output_dir=project_root() / "output",
    )


def default_hitran_dir() -> Path:
    """Return the default HITRAN cache directory inside the project root."""
    return project_root() / "hitran"


def parse_wn_range(value: str) -> tuple[float, float]:
    """Parse MIN,MAX wavenumber bounds."""
    lower_text, sep, upper_text = str(value).partition(",")
    if not sep:
        raise argparse.ArgumentTypeError("wn-range must have the form min,max")
    try:
        lower = float(lower_text)
        upper = float(upper_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("wn-range must contain numeric min,max values") from exc
    if upper < lower:
        raise argparse.ArgumentTypeError("wn-range max must be >= min")
    return lower, upper


def build_band(args: argparse.Namespace):
    """Build the standard single-state spectral band config."""
    from .config import SpectralBandConfig

    wn_min, wn_max = args.wn_range
    return SpectralBandConfig("single_state", wn_min, wn_max, args.resolution)


def default_cli_output_path(args: argparse.Namespace, *, suffix: str) -> Path:
    """Return the default named output path for a CLI command."""
    return default_named_output_path(
        target_name=args.species,
        plot_type=args.command,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        wn_range=args.wn_range,
        suffix=suffix,
        output_dir=project_root() / "output",
    )


def add_single_state_arguments(parser: argparse.ArgumentParser, *, include_path_length: bool = False) -> None:
    """Add shared single-state spectroscopy CLI arguments."""
    parser.add_argument("--output", type=Path, default=None, metavar="PATH", help="Output NetCDF path. Defaults to an auto-generated path under output/.")
    parser.add_argument("--hitran-dir", type=Path, default=default_hitran_dir(), metavar="DIR", help="Directory for downloaded HITRAN line data.")
    parser.add_argument("--species", default="CO2", metavar="NAME", help="Molecular target, for example CO2, H2O, CH4, NH3, H2S, H2, or N2.")
    parser.add_argument("--refresh-hitran", action="store_true", help="Re-download HITRAN line tables even if cached.")
    parser.add_argument(
        "--broadening-composition",
        default=None,
        metavar="BROADENER:FRACTION,...",
        help="Line-broadening gas composition, for example air:0.8,self:0.2 or H2:0.85,He:0.15.",
    )
    parser.add_argument("--temperature-k", type=float, default=300.0, metavar="K", help="Gas temperature in kelvin.")
    parser.add_argument("--pressure-bar", type=float, default=1.0, metavar="BAR", help="Gas pressure in bar.")
    if include_path_length:
        parser.add_argument("--path-length-m", type=float, default=1.0, metavar="M", help="Propagation path length in meters.")
    parser.add_argument("--wn-range", type=parse_wn_range, default=(20.0, 2500.0), metavar="MIN,MAX", help="Wavenumber range in cm^-1.")
    parser.add_argument("--resolution", type=float, default=1.0, metavar="CM^-1", help="Wavenumber grid spacing in cm^-1.")
