"""Shared CLI helpers for CIA plotting scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import resolve_hitran_cia_pair
from .hitran_cia import (
    load_cia_dataset,
    plot_cia_attenuation_coefficient,
    plot_cia_cross_section,
    plot_cia_transmission,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"))
    parser.add_argument("--filename", default=None)
    parser.add_argument("--pair", default="H2-H2")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--wn-min", type=float, default=20.0)
    parser.add_argument("--wn-max", type=float, default=10000.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--refresh", action="store_true")


def _validate_and_build_grid(args: argparse.Namespace) -> np.ndarray:
    if args.resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if args.wn_max < args.wn_min:
        raise ValueError("wn-max must be >= wn-min")
    count = int(round((args.wn_max - args.wn_min) / args.resolution))
    return args.wn_min + np.arange(count + 1, dtype=np.float64) * args.resolution


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
