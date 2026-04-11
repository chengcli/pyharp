"""Shared CLI helpers for CIA plotting scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SpectralBandConfig, resolve_hitran_cia_pair
from .hitran_cia import (
    load_cia_dataset,
    plot_cia_attenuation_coefficient,
    plot_cia_cross_section,
    plot_cia_transmission,
)


def _parse_wn_range(value: str) -> tuple[float, float]:
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


def _wn_bounds(args: argparse.Namespace) -> tuple[float, float]:
    return tuple(args.wn_range)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"))
    parser.add_argument("--filename", default=None)
    parser.add_argument("--pair", default="H2-H2")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--wn-range", type=_parse_wn_range, default=(20.0, 10000.0))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--refresh", action="store_true")


def _validate_and_build_grid(args: argparse.Namespace) -> np.ndarray:
    wn_min, wn_max = _wn_bounds(args)
    band = SpectralBandConfig(
        name="cia_plot",
        wavenumber_min_cm1=wn_min,
        wavenumber_max_cm1=wn_max,
        resolution_cm1=args.resolution,
    )
    return band.grid()


def _resolve_filename(args: argparse.Namespace) -> str:
    if args.filename:
        return str(args.filename)
    return resolve_hitran_cia_pair(args.pair).filename


def _load_dataset(args: argparse.Namespace):
    return load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=_resolve_filename(args),
        pair=args.pair,
        refresh=args.refresh,
    )


def _print_summary(dataset, grid: np.ndarray, label: str, values: np.ndarray, unit: str = "") -> None:
    print(f"CIA file: {dataset.source_path}")
    print(
        f"Temperatures in file: {dataset.temperatures_k.min():.1f}.."
        f"{dataset.temperatures_k.max():.1f} K ({dataset.temperatures_k.size} sets)"
    )
    print(f"Grid points: {grid.size}")
    suffix = f" {unit}" if unit else ""
    print(f"{label} min/max: {values.min():.3e} .. {values.max():.3e}{suffix}")


def build_binary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and plot a HITRAN CIA binary absorption coefficient spectrum.")
    _add_common_arguments(parser)
    parser.add_argument("--figure", type=Path, default=Path("output/h2_h2_cia_300K.png"))
    return parser


def main_binary() -> None:
    args = build_binary_parser().parse_args()
    run_binary(args)


def run_binary(args: argparse.Namespace) -> None:
    grid = _validate_and_build_grid(args)
    dataset = _load_dataset(args)
    values = plot_cia_cross_section(
        dataset,
        temperature_k=args.temperature_k,
        wavenumber_grid_cm1=grid,
        figure_path=args.figure,
    )
    positive = values[values > 0.0]
    _print_summary(dataset, grid, "Coefficient", positive, "cm^5 / molecule^2")


def build_attenuation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and plot a HITRAN CIA attenuation coefficient spectrum in 1/m.")
    _add_common_arguments(parser)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--figure", type=Path, default=Path("output/h2_h2_cia_attenuation_300K_1bar.png"))
    return parser


def main_attenuation() -> None:
    args = build_attenuation_parser().parse_args()
    run_attenuation(args)


def run_attenuation(args: argparse.Namespace) -> None:
    grid = _validate_and_build_grid(args)
    dataset = _load_dataset(args)
    values = plot_cia_attenuation_coefficient(
        dataset,
        temperature_k=args.temperature_k,
        pressure_pa=args.pressure_bar * 1.0e5,
        wavenumber_grid_cm1=grid,
        figure_path=args.figure,
    )
    positive = values[values > 0.0]
    _print_summary(dataset, grid, "Attenuation", positive, "1/m")


def build_transmission_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and plot a HITRAN CIA transmission spectrum over a fixed path length.")
    _add_common_arguments(parser)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--path-length-km", type=float, default=1.0)
    parser.add_argument("--figure", type=Path, default=Path("output/h2_h2_cia_transmission_300K_1bar_1km.png"))
    return parser


def main_transmission() -> None:
    args = build_transmission_parser().parse_args()
    run_transmission(args)


def run_transmission(args: argparse.Namespace) -> None:
    if args.path_length_km <= 0.0:
        raise ValueError("path-length-km must be positive")
    grid = _validate_and_build_grid(args)
    dataset = _load_dataset(args)
    values = plot_cia_transmission(
        dataset,
        temperature_k=args.temperature_k,
        pressure_pa=args.pressure_bar * 1.0e5,
        path_length_m=args.path_length_km * 1000.0,
        wavenumber_grid_cm1=grid,
        figure_path=args.figure,
    )
    _print_summary(dataset, grid, "Transmission", values)
