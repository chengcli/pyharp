"""Command-line interface for single-state spectroscopy utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SpectroscopyConfig, SpectralBandConfig
from .spectrum import compute_absorption_spectrum, plot_absorption_spectrum, write_spectrum_dataset
from .transmittance import (
    compute_transmittance_from_config,
    plot_transmittance_spectrum,
    write_transmittance_dataset,
)


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def default_output_path() -> Path:
    """Return the default NetCDF output path inside the project root."""
    return project_root() / "output" / "co2_absorption_300K_1bar.nc"


def default_hitran_dir() -> Path:
    """Return the default HITRAN cache directory inside the project root."""
    return project_root() / "hitran"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    spectrum = subparsers.add_parser("spectrum", help="Compute and plot absorption at one pressure-temperature state.")
    spectrum.add_argument("--output", type=Path, default=default_output_path())
    spectrum.add_argument("--figure", type=Path, default=project_root() / "output" / "co2_absorption_300K_1bar.png")
    spectrum.add_argument("--hitran-dir", type=Path, default=default_hitran_dir())
    spectrum.add_argument("--species", default="CO2")
    spectrum.add_argument("--refresh-hitran", action="store_true")
    spectrum.add_argument("--temperature-k", type=float, default=300.0)
    spectrum.add_argument("--pressure-bar", type=float, default=1.0)
    spectrum.add_argument("--wn-min", type=float, default=20.0)
    spectrum.add_argument("--wn-max", type=float, default=2500.0)
    spectrum.add_argument("--resolution", type=float, default=1.0)

    transmittance = subparsers.add_parser("transmittance", help="Compute and plot transmittance at one pressure-temperature state.")
    transmittance.add_argument("--output", type=Path, default=project_root() / "output" / "co2_transmittance_300K_1bar_1m.nc")
    transmittance.add_argument("--figure", type=Path, default=project_root() / "output" / "co2_transmittance_300K_1bar_1m.png")
    transmittance.add_argument("--hitran-dir", type=Path, default=default_hitran_dir())
    transmittance.add_argument("--species", default="CO2")
    transmittance.add_argument("--refresh-hitran", action="store_true")
    transmittance.add_argument("--temperature-k", type=float, default=300.0)
    transmittance.add_argument("--pressure-bar", type=float, default=1.0)
    transmittance.add_argument("--path-length-m", type=float, default=1.0)
    transmittance.add_argument("--wn-min", type=float, default=20.0)
    transmittance.add_argument("--wn-max", type=float, default=2500.0)
    transmittance.add_argument("--resolution", type=float, default=1.0)
    return parser


def main() -> None:
    """Run the CLI."""
    args = build_parser().parse_args()
    if args.command == "spectrum":
        band = SpectralBandConfig("single_state", args.wn_min, args.wn_max, args.resolution)
        config = SpectroscopyConfig(
            output_path=args.output,
            hitran_cache_dir=args.hitran_dir,
            species_name=args.species,
            refresh_hitran=args.refresh_hitran,
        )
        spectrum = compute_absorption_spectrum(
            config=config,
            band=band,
            temperature_k=args.temperature_k,
            pressure_pa=args.pressure_bar * 1.0e5,
        )
        write_spectrum_dataset(spectrum, args.output)
        plot_absorption_spectrum(spectrum, args.figure)
        return
    if args.command == "transmittance":
        band = SpectralBandConfig("single_state", args.wn_min, args.wn_max, args.resolution)
        config = SpectroscopyConfig(
            output_path=args.output,
            hitran_cache_dir=args.hitran_dir,
            species_name=args.species,
            refresh_hitran=args.refresh_hitran,
        )
        transmittance = compute_transmittance_from_config(
            config=config,
            band=band,
            temperature_k=args.temperature_k,
            pressure_pa=args.pressure_bar * 1.0e5,
            path_length_m=args.path_length_m,
        )
        write_transmittance_dataset(transmittance, args.output)
        plot_transmittance_spectrum(transmittance, args.figure)
        return


if __name__ == "__main__":
    main()
