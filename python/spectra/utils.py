"""Shared helpers for spectroscopy command-line interfaces."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import sys

from .config import SpectralBandConfig


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


def _format_value(value: float | int | str, unit: str = "") -> str:
    if isinstance(value, str):
        text = value
    else:
        text = f"{float(value):g}"
    return f"{text.replace('-', 'm').replace('.', 'p')}{unit}"


def _clean_token(value: object) -> str:
    text = str(value).strip().lower()
    pieces: list[str] = []
    previous_was_separator = False
    for char in text:
        if char.isalnum():
            pieces.append(char)
            previous_was_separator = False
        elif char == ".":
            pieces.append("p")
            previous_was_separator = False
        elif not previous_was_separator:
            pieces.append("_")
            previous_was_separator = True
    return "".join(pieces).strip("_") or "output"


def default_output_path(
    *,
    target_name: object = "CO2",
    plot_type: str = "xsection",
    temperature_k: float | str = 300.0,
    pressure_bar: float | str = 1.0,
    wn_range: tuple[float, float] = (20.0, 2500.0),
    suffix: str = ".nc",
    output_dir: Path | None = None,
) -> Path:
    """Return a default spectroscopy CLI output path."""
    if output_dir is None:
        output_dir = project_root() / "output"
    wn_min, wn_max = wn_range
    stem = "_".join(
        [
            _clean_token(target_name),
            _clean_token(plot_type),
            _format_value(temperature_k, "K"),
            _format_value(pressure_bar, "bar"),
            _format_value(wn_min),
            _format_value(wn_max, "cm1"),
        ]
    )
    return Path(output_dir) / f"{stem}{suffix}"


def default_hitran_dir() -> Path:
    """Return the default HITRAN cache directory in the current working directory."""
    return Path("hitran")


def default_orton_xiz_cia_dir() -> Path:
    """Return the default legacy Orton/Xiz CIA cache directory."""
    return Path("orton_xiz_cia")


def process_pool_context() -> mp.context.BaseContext:
    """Return a process start context that is safe for the current platform."""
    if sys.platform == "darwin":
        return mp.get_context("spawn")
    start_methods = mp.get_all_start_methods()
    if "fork" in start_methods:
        return mp.get_context("fork")
    current = mp.get_start_method(allow_none=True)
    if current is not None:
        return mp.get_context(current)
    return mp.get_context(start_methods[0])


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


def build_band_from_range(
    wn_range: tuple[float, float],
    resolution_cm1: float,
    *,
    name: str = "single_state",
) -> SpectralBandConfig:
    """Build a validated regular spectral band from explicit bounds."""
    wn_min, wn_max = wn_range
    if resolution_cm1 <= 0.0:
        raise ValueError("resolution must be positive")
    if wn_max < wn_min:
        raise ValueError("wn-range max must be >= min")
    return SpectralBandConfig(name, float(wn_min), float(wn_max), float(resolution_cm1))


def build_band(args: argparse.Namespace):
    """Build the standard single-state spectral band config."""
    return build_band_from_range(tuple(args.wn_range), float(args.resolution))


def build_grid(args: argparse.Namespace, *, name: str = "single_state"):
    """Build a validated spectral grid from CLI-style args."""
    return build_band_from_range(tuple(args.wn_range), float(args.resolution), name=name).grid()


def default_cli_output_path(args: argparse.Namespace, *, suffix: str) -> Path:
    """Return the default named output path for a CLI command."""
    return default_output_path(
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
