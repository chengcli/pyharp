"""Plotting and CLI helpers for HITRAN collision-induced absorption."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .blackbody import compute_normalized_blackbody_curve
from .config import cia_database_for_model, resolve_hitran_cia_filename
from .hitran_cia_utils import (
    CiaDataset,
    compute_cia_attenuation_m1,
    compute_cia_transmission,
    load_cia_dataset,
)
from .orton_xiz_cia import load_orton_xiz_cia_dataset, resolve_orton_xiz_cia_filename
from .utils import build_grid, default_hitran_dir, default_orton_xiz_cia_dir, parse_wn_range


def plot_cia_cross_section(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    wavenumber_grid_cm1: np.ndarray,
    figure_path: Path,
) -> np.ndarray:
    """Plot a CIA binary absorption coefficient spectrum in cm^5 molecule^-2."""
    import matplotlib.pyplot as plt

    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    binary_xsec = dataset.interpolate_to_grid(temperature_k, wavenumber_grid_cm1)
    positive = binary_xsec[binary_xsec > 0.0]
    if positive.size == 0:
        raise ValueError("No positive CIA coefficients were found on the requested grid.")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavenumber_grid_cm1, binary_xsec, color="tab:blue", linewidth=1.25)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Binary absorption coefficient [cm$^5$ / molecule$^2$]")
    ax.set_title(f"{dataset.pair} CIA at {temperature_k:.1f} K")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return binary_xsec


def plot_cia_attenuation_coefficient(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    pressure_pa: float,
    wavenumber_grid_cm1: np.ndarray,
    figure_path: Path,
) -> np.ndarray:
    """Plot a CIA attenuation coefficient spectrum in 1/m."""
    import matplotlib.pyplot as plt

    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    attenuation_m1 = compute_cia_attenuation_m1(
        dataset,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        wavenumber_grid_cm1=wavenumber_grid_cm1,
    )
    positive = attenuation_m1[attenuation_m1 > 0.0]
    if positive.size == 0:
        raise ValueError("No positive CIA attenuation coefficients were found on the requested grid.")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavenumber_grid_cm1, attenuation_m1, color="tab:orange", linewidth=1.25)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Attenuation coefficient [1/m]")
    ax.set_title(f"{dataset.pair} CIA attenuation at {temperature_k:.1f} K and {pressure_pa / 1.0e5:.3f} bar")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return attenuation_m1


def plot_cia_transmission(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    pressure_pa: float,
    path_length_m: float,
    wavenumber_grid_cm1: np.ndarray,
    figure_path: Path,
) -> np.ndarray:
    """Plot CIA transmission over a fixed path length."""
    import matplotlib.pyplot as plt

    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    transmission = compute_cia_transmission(
        dataset,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        path_length_m=path_length_m,
        wavenumber_grid_cm1=wavenumber_grid_cm1,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavenumber_grid_cm1, transmission, color="tab:green", linewidth=1.25)
    ax.plot(
        wavenumber_grid_cm1,
        compute_normalized_blackbody_curve(
            wavenumber_cm1=wavenumber_grid_cm1,
            temperature_k=temperature_k,
        ),
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Blackbody",
    )
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Transmission")
    ax.set_ylim(0.0, 1.01)
    ax.set_title(
        f"{dataset.pair} CIA transmission at {temperature_k:.1f} K, "
        f"{pressure_pa / 1.0e5:.3f} bar, L={path_length_m / 1000.0:.3f} km"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return transmission


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hitran-dir", type=Path, default=default_hitran_dir())
    parser.add_argument("--cia-dir", type=Path, default=None)
    parser.add_argument("--cia-model", choices=("auto", "2011", "2018", "xiz", "orton"), default="auto")
    parser.add_argument("--h2-state", choices=("eq", "nm"), default="eq")
    parser.add_argument("--filename", default=None)
    parser.add_argument("--pair", default="H2-H2")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--wn-range", type=parse_wn_range, default=(20.0, 10000.0))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--refresh", action="store_true")


def _resolve_filename(args: argparse.Namespace) -> str:
    if args.filename:
        return str(args.filename)
    if cia_database_for_model(args.cia_model) == "orton_xiz":
        return resolve_orton_xiz_cia_filename(pair=args.pair, model=args.cia_model, state=args.h2_state)
    return resolve_hitran_cia_filename(pair=args.pair, model=args.cia_model, state=args.h2_state).filename


def _load_dataset(args: argparse.Namespace):
    if cia_database_for_model(args.cia_model) == "orton_xiz":
        return load_orton_xiz_cia_dataset(
            cache_dir=args.cia_dir or default_orton_xiz_cia_dir(),
            pair=args.pair,
            model=args.cia_model,
            state=args.h2_state,
            refresh=args.refresh,
        )
    return load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=_resolve_filename(args),
        pair=args.pair,
        refresh=args.refresh,
    )


def _print_summary(dataset, grid: np.ndarray, label: str, values: np.ndarray, unit: str = "") -> None:
    print(f"CIA file: {dataset.source_path}")
    temperatures = np.asarray(dataset.temperatures_k, dtype=np.float64)
    print(f"Temperatures in file: {temperatures.min():.1f}..{temperatures.max():.1f} K ({temperatures.size} sets)")
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
    grid = build_grid(args, name="cia_plot")
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
    grid = build_grid(args, name="cia_plot")
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
    grid = build_grid(args, name="cia_plot")
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
