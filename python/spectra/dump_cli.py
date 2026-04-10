"""NetCDF dump CLI for single-state spectroscopy utilities."""

from __future__ import annotations

import argparse
from textwrap import dedent

from .config import SpectroscopyConfig, parse_broadening_composition
from .hitran_lines import build_line_provider, download_hitran_lines
from .shared_cli import (
    HelpFormatter,
    add_single_state_arguments,
    build_band,
    default_cli_output_path,
)
from .spectrum import compute_absorption_spectrum, write_spectrum_dataset
from .transmittance import compute_transmittance_spectrum, write_transmittance_dataset


def build_parser() -> argparse.ArgumentParser:
    """Create the dump CLI parser."""
    parser = argparse.ArgumentParser(
        description="Write single-state spectroscopy NetCDF products from HITRAN line data.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump spectrum --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump spectrum --species NH3 --broadening-composition H2:0.9,He:0.1 --wn-range=20,2500
              pyharp-dump transmittance --species H2O --path-length-m 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

            Run "pyharp-dump COMMAND -h" for command-specific options.
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    spectrum = subparsers.add_parser(
        "spectrum",
        help="Write a single-state absorption dataset.",
        description="Compute molecular absorption at one pressure-temperature state and write a NetCDF dataset.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump spectrum --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump spectrum --species H2O --broadening-composition air:0.8,self:0.2 --wn-range=1000,1500
            """
        ),
    )
    add_single_state_arguments(spectrum)

    transmittance = subparsers.add_parser(
        "transmittance",
        help="Write a single-state transmittance dataset.",
        description="Compute molecular transmittance at one pressure-temperature state and write a NetCDF dataset.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump transmittance --species CO2 --path-length-m 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump transmittance --species CH4 --path-length-m 10 --broadening-composition H2:0.85,He:0.15 --wn-range=1000,4000
            """
        ),
    )
    add_single_state_arguments(transmittance, include_path_length=True)
    return parser


def main() -> None:
    """Run the dump CLI."""
    args = build_parser().parse_args()
    if args.command == "spectrum":
        band = build_band(args)
        output_path = args.output or default_cli_output_path(args, suffix=".nc")
        config = SpectroscopyConfig(
            output_path=output_path,
            hitran_cache_dir=args.hitran_dir,
            species_name=args.species,
            broadening_composition=parse_broadening_composition(args.broadening_composition),
            refresh_hitran=args.refresh_hitran,
        )
        line_db = download_hitran_lines(config, band)
        line_provider = build_line_provider(config, line_db)
        spectrum = compute_absorption_spectrum(
            config=config,
            band=band,
            temperature_k=args.temperature_k,
            pressure_pa=args.pressure_bar * 1.0e5,
            line_db=line_db,
        )
        write_spectrum_dataset(spectrum, output_path)
        print(f"Wrote NetCDF: {output_path}")
        print(f"Broadening: {line_provider.broadening_summary()}")
        return
    if args.command == "transmittance":
        band = build_band(args)
        output_path = args.output or default_cli_output_path(args, suffix=".nc")
        config = SpectroscopyConfig(
            output_path=output_path,
            hitran_cache_dir=args.hitran_dir,
            species_name=args.species,
            broadening_composition=parse_broadening_composition(args.broadening_composition),
            refresh_hitran=args.refresh_hitran,
        )
        line_db = download_hitran_lines(config, band)
        line_provider = build_line_provider(config, line_db)
        spectrum = compute_absorption_spectrum(
            config=config,
            band=band,
            temperature_k=args.temperature_k,
            pressure_pa=args.pressure_bar * 1.0e5,
            line_db=line_db,
        )
        transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=args.path_length_m)
        write_transmittance_dataset(transmittance, output_path)
        print(f"Wrote NetCDF: {output_path}")
        print(f"Broadening: {line_provider.broadening_summary()}")
        return
