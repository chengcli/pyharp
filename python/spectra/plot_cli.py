"""Unified plotting command for spectroscopy diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

from . import atm_overview_cli, cia_plot_cli, molecule_plot_cli
from .output_names import default_output_path


class _SplitSpeciesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        species: list[str] = []
        for value in values:
            species.extend(part.strip() for part in str(value).split(",") if part.strip())
        setattr(namespace, self.dest, species)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
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


def _add_state_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature-k", type=float, default=300.0, metavar="K", help="Gas temperature in kelvin.")
    parser.add_argument("--pressure-bar", type=float, default=1.0, metavar="BAR", help="Gas pressure in bar.")


def _add_common_arguments(parser: argparse.ArgumentParser, *, allow_multiple_ranges: bool = False) -> None:
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"), metavar="DIR", help="Directory for downloaded HITRAN line and CIA data.")
    parser.add_argument("--resolution", type=float, default=1.0, metavar="CM^-1", help="Wavenumber grid spacing in cm^-1.")
    parser.add_argument(
        "--broadening-composition",
        default=None,
        metavar="BROADENER:FRACTION,...",
        help="Line-broadening gas composition for molecular line calculations, for example air:0.8,self:0.2 or H2:0.85,He:0.15.",
    )
    if allow_multiple_ranges:
        parser.add_argument(
            "--wn-range",
            dest="wn_ranges",
            action="append",
            type=molecule_plot_cli._parse_wn_range,
            metavar="MIN,MAX",
            help="Wavenumber range in cm^-1. Repeat for multi-page overview PDFs.",
        )
    else:
        parser.add_argument(
            "--wn-range",
            type=molecule_plot_cli._parse_wn_range,
            default=None,
            metavar="MIN,MAX",
            help="Wavenumber range in cm^-1. CIA plots default to 20,10000; molecular and mixture plots default to 20,2500.",
        )


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--refresh-hitran", action="store_true", help="Re-download HITRAN line tables even if cached.")
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/", metavar="URL", help="HITRAN CIA index URL used to resolve CIA files.")
    parser.add_argument("--refresh-cia", action="store_true", help="Re-download HITRAN CIA files even if cached.")


def _add_selector_arguments(parser: argparse.ArgumentParser, *, include_composition: bool = False) -> None:
    parser.add_argument("--pair", default=None, metavar="PAIR", help="CIA pair target, for example H2-H2 or H2-He.")
    parser.add_argument("--species", default=None, metavar="NAME", help="Molecular target, for example H2O, CO2, CH4, H2, or N2.")
    if include_composition:
        parser.add_argument("--composition", default=None, metavar="SPECIES:FRACTION,...", help="Gas mixture target, for example H2O:0.1,H2:0.9.")


def _validate_single_selector(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    selectors = [bool(getattr(args, "pair", None)), bool(getattr(args, "species", None)), bool(getattr(args, "composition", None))]
    if sum(selectors) > 1:
        parser.error("choose only one of --pair, --species, or --composition")


def _combined_range(wn_ranges: list[tuple[float, float]]) -> tuple[float, float]:
    return min(wn_min for wn_min, _ in wn_ranges), max(wn_max for _, wn_max in wn_ranges)


def _default_figure(
    *,
    target_name: object,
    plot_type: str,
    temperature_k: float,
    pressure_bar: float,
    wn_range: tuple[float, float],
    suffix: str = ".png",
) -> Path:
    return default_output_path(
        target_name=target_name,
        plot_type=plot_type,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        wn_range=wn_range,
        suffix=suffix,
    )


def _as_cia_args(args: argparse.Namespace, *, default_pair: str, plot_type: str) -> argparse.Namespace:
    wn_range = args.wn_range or (20.0, 10000.0)
    pressure_bar = getattr(args, "pressure_bar", 1.0)
    pair = args.pair or default_pair
    refresh_cia = getattr(args, "refresh_cia", getattr(args, "refresh", False))
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        filename=getattr(args, "filename", None),
        pair=pair,
        temperature_k=args.temperature_k,
        wn_range=wn_range,
        resolution=args.resolution,
        refresh=refresh_cia,
        refresh_cia=refresh_cia,
        cia_index_url=getattr(args, "cia_index_url", None),
        pressure_bar=pressure_bar,
        path_length_km=getattr(args, "path_length_km", 1.0),
        figure=args.figure
        or _default_figure(
            target_name=pair,
            plot_type=plot_type,
            temperature_k=args.temperature_k,
            pressure_bar=pressure_bar,
            wn_range=wn_range,
        ),
    )


def _as_molecule_args(args: argparse.Namespace, *, plot_type: str) -> argparse.Namespace:
    species = args.species or "H2O"
    wn_range = args.wn_range or (20.0, 2500.0)
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        species=species,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        wn_range=wn_range,
        resolution=args.resolution,
        refresh_hitran=args.refresh_hitran,
        cia_filename=getattr(args, "cia_filename", None),
        cia_pair=getattr(args, "cia_pair", None),
        cia_index_url=args.cia_index_url,
        refresh_cia=args.refresh_cia,
        broadening_composition=getattr(args, "broadening_composition", None),
        path_length_km=getattr(args, "path_length_km", 1.0),
        figure=args.figure
        or _default_figure(
            target_name=species,
            plot_type=plot_type,
            temperature_k=args.temperature_k,
            pressure_bar=args.pressure_bar,
            wn_range=wn_range,
        ),
    )


def _as_molecule_overview_args(args: argparse.Namespace, *, species: str, wn_range: tuple[float, float]) -> argparse.Namespace:
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        species=species,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        wn_range=wn_range,
        resolution=args.resolution,
        refresh_hitran=args.refresh_hitran,
        cia_filename=getattr(args, "cia_filename", None),
        cia_pair=getattr(args, "cia_pair", None),
        cia_index_url=args.cia_index_url,
        refresh_cia=args.refresh_cia,
        broadening_composition=getattr(args, "broadening_composition", None),
        path_length_km=args.path_length_km,
        figure=args.figure
        or _default_figure(
            target_name=species,
            plot_type="overview",
            temperature_k=args.temperature_k,
            pressure_bar=args.pressure_bar,
            wn_range=wn_range,
            suffix=".pdf",
        ),
    )


def _as_molecule_overview_batch_args(args: argparse.Namespace, *, species: list[str], wn_ranges: list[tuple[float, float]]) -> argparse.Namespace:
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        species=species,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        resolution=args.resolution,
        path_length_km=args.path_length_km,
        wn_ranges=wn_ranges,
        refresh_hitran=args.refresh_hitran,
        cia_index_url=args.cia_index_url,
        refresh_cia=args.refresh_cia,
        broadening_composition=getattr(args, "broadening_composition", None),
        figure=args.figure
        or _default_figure(
            target_name="_".join(species),
            plot_type="overview",
            temperature_k=args.temperature_k,
            pressure_bar=args.pressure_bar,
            wn_range=_combined_range(wn_ranges),
            suffix=".pdf",
        ),
    )


def _as_atm_overview_args(args: argparse.Namespace, *, wn_ranges: list[tuple[float, float]]) -> argparse.Namespace:
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        composition=args.composition,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        path_length_km=args.path_length_km,
        resolution=args.resolution,
        wn_ranges=wn_ranges,
        cia_index_url=args.cia_index_url,
        refresh_hitran=args.refresh_hitran,
        refresh_cia=args.refresh_cia,
        broadening_composition=getattr(args, "broadening_composition", None),
        figure=args.figure
        or _default_figure(
            target_name=args.composition,
            plot_type="overview",
            temperature_k=args.temperature_k,
            pressure_bar=args.pressure_bar,
            wn_range=_combined_range(wn_ranges),
            suffix=".pdf",
        ),
        manifest=args.manifest,
    )


def _as_atm_args(args: argparse.Namespace, *, plot_type: str, wn_range: tuple[float, float]) -> argparse.Namespace:
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        composition=args.composition,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        path_length_km=getattr(args, "path_length_km", 1.0),
        resolution=args.resolution,
        cia_index_url=args.cia_index_url,
        refresh_hitran=args.refresh_hitran,
        refresh_cia=args.refresh_cia,
        broadening_composition=getattr(args, "broadening_composition", None),
        figure=args.figure
        or _default_figure(
            target_name=args.composition,
            plot_type=plot_type,
            temperature_k=args.temperature_k,
            pressure_bar=args.pressure_bar,
            wn_range=wn_range,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot pyharp spectroscopy diagnostics.",
        formatter_class=_HelpFormatter,
        epilog=dedent(
            """\
            Target selection:
              --pair selects a HITRAN CIA pair.
              --species selects a molecule.
              --composition selects a gas mixture.

            Examples:
              pyharp-plot binary --pair H2-H2 --temperature-k 300 --wn-range=20,10000
              pyharp-plot xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-plot transmission --composition H2O:0.1,H2:0.9 --path-length-km 1 --wn-range=25,2500

            Run "pyharp-plot COMMAND -h" for command-specific options.
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    binary = subparsers.add_parser(
        "binary",
        help="Plot a HITRAN CIA binary coefficient.",
        description="Plot a HITRAN CIA binary absorption coefficient spectrum.",
        formatter_class=_HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-plot binary --pair H2-H2 --temperature-k 300 --wn-range=20,10000
              pyharp-plot binary --pair H2-He --temperature-k 500 --resolution 5 --figure output/h2_he_cia.png
            """
        ),
    )
    _add_common_arguments(binary)
    binary.add_argument("--pair", default="H2-H2", metavar="PAIR", help="CIA pair target, for example H2-H2 or H2-He.")
    binary.add_argument("--filename", default=None, metavar="FILE", help="Use a specific CIA filename instead of resolving one from --pair.")
    binary.add_argument("--temperature-k", type=float, default=300.0, metavar="K", help="Gas temperature in kelvin.")
    binary.add_argument("--refresh", action="store_true", help="Re-download the CIA file even if cached.")
    binary.add_argument("--figure", type=Path, default=None, metavar="PATH", help="Output PNG path. Defaults to an auto-generated path under output/.")

    xsection = subparsers.add_parser(
        "xsection",
        help="Plot a molecular absorption cross section.",
        description="Plot molecular absorption cross section at one pressure-temperature state.",
        formatter_class=_HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-plot xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-plot xsection --species CH4 --temperature-k 700 --pressure-bar 0.1 --wn-range=1000,4000 --refresh-hitran
            """
        ),
    )
    _add_common_arguments(xsection)
    _add_state_arguments(xsection)
    xsection.add_argument("--species", default="H2O", metavar="NAME", help="Molecular target, for example H2O, CO2, CH4, H2, or N2.")
    _add_cache_arguments(xsection)
    xsection.add_argument("--cia-filename", default=None, metavar="FILE", help="Optional CIA filename to include as the secondary continuum source.")
    xsection.add_argument("--cia-pair", default=None, metavar="PAIR", help="Optional CIA pair to resolve as the secondary continuum source.")
    xsection.add_argument("--figure", type=Path, default=None, metavar="PATH", help="Output PNG path. Defaults to an auto-generated path under output/.")

    for name in ("attenuation", "transmission"):
        path_help = "Transmission path length in kilometers." if name == "transmission" else None
        examples = (
            f"  pyharp-plot {name} --pair H2-H2 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000\n"
            f"  pyharp-plot {name} --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500\n"
            f"  pyharp-plot {name} --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --wn-range=25,2500"
        )
        if name == "transmission":
            examples = examples.replace("--wn-range=20,10000", "--path-length-km 1 --wn-range=20,10000")
            examples = examples.replace("--wn-range=20,2500", "--path-length-km 0.5 --wn-range=20,2500")
            examples = examples.replace("--wn-range=25,2500", "--path-length-km 1 --wn-range=25,2500")
        subparser = subparsers.add_parser(
            name,
            help=f"Plot {name} for a CIA pair, molecule, or gas mixture.",
            description=f"Plot {name} for exactly one target: --pair, --species, or --composition.",
            formatter_class=_HelpFormatter,
            epilog=f"Examples:\n{examples}",
        )
        _add_common_arguments(subparser)
        _add_state_arguments(subparser)
        _add_selector_arguments(subparser, include_composition=True)
        _add_cache_arguments(subparser)
        subparser.add_argument("--filename", default=None, metavar="FILE", help="Use a specific CIA filename for --pair targets.")
        subparser.add_argument("--cia-filename", default=None, metavar="FILE", help="Optional CIA filename to include for molecular targets.")
        subparser.add_argument("--cia-pair", default=None, metavar="PAIR", help="Optional CIA pair to include for molecular targets.")
        if name == "transmission":
            subparser.add_argument("--path-length-km", type=float, default=1.0, metavar="KM", help=path_help)
        subparser.add_argument("--figure", type=Path, default=None, metavar="PATH", help="Output PNG path. Defaults to an auto-generated path under output/.")

    overview = subparsers.add_parser(
        "overview",
        help="Generate molecule or atmosphere overview PDFs.",
        description="Generate molecule or atmosphere overview plots. Use either --species or --composition.",
        formatter_class=_HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-plot overview --species H2O --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=20,2500
              pyharp-plot overview --species H2O CO2 CH4 --temperature-k 500 --pressure-bar 0.5 --wn-range=25,2500 --wn-range=2500,10000
              pyharp-plot overview --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --wn-range=25,2500 --manifest output/h2o_h2_sources.json
            """
        ),
    )
    _add_common_arguments(overview, allow_multiple_ranges=True)
    _add_state_arguments(overview)
    overview.add_argument("--path-length-km", type=float, default=1.0, metavar="KM", help="Transmission path length in kilometers.")
    overview.add_argument(
        "--species",
        nargs="+",
        action=_SplitSpeciesAction,
        default=None,
        metavar="NAME",
        help="One or more molecular targets. Accepts space-separated or comma-separated names.",
    )
    overview.add_argument("--composition", default=None, metavar="SPECIES:FRACTION,...", help="Gas mixture target, for example H2O:0.1,H2:0.9.")
    _add_cache_arguments(overview)
    overview.add_argument("--cia-filename", default=None, metavar="FILE", help="Optional CIA filename to include for molecular overview pages.")
    overview.add_argument("--cia-pair", default=None, metavar="PAIR", help="Optional CIA pair to include for molecular overview pages.")
    overview.add_argument("--figure", type=Path, default=None, metavar="PATH", help="Output PDF path. Defaults to an auto-generated path under output/.")
    overview.add_argument("--manifest", type=Path, default=None, metavar="PATH", help="Output manifest JSON path for composition overview PDFs.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "binary":
        cia_plot_cli.run_binary(_as_cia_args(args, default_pair="H2-H2", plot_type="binary"))
        return

    if args.command == "xsection":
        molecule_plot_cli.run_xsection(_as_molecule_args(args, plot_type="xsection"))
        return

    if args.command in {"attenuation", "transmission"}:
        _validate_single_selector(args, parser)
        if args.composition:
            wn_range = args.wn_range or (20.0, 2500.0)
            atm_args = _as_atm_args(args, plot_type=args.command, wn_range=wn_range)
            if args.command == "attenuation":
                atm_overview_cli.run_atm_attenuation(atm_args, wn_range=wn_range)
            else:
                atm_overview_cli.run_atm_transmission(atm_args, wn_range=wn_range)
            return
        if args.pair:
            cia_args = _as_cia_args(args, default_pair=args.pair, plot_type=args.command)
            if args.command == "attenuation":
                cia_plot_cli.run_attenuation(cia_args)
            else:
                cia_plot_cli.run_transmission(cia_args)
            return

        molecule_args = _as_molecule_args(args, plot_type=args.command)
        if args.command == "attenuation":
            molecule_plot_cli.run_attenuation(molecule_args)
        else:
            molecule_plot_cli.run_transmission(molecule_args)
        return

    if args.command == "overview":
        if args.composition and args.species:
            parser.error("choose only one of --composition or --species")
        wn_ranges = args.wn_ranges or [(20.0, 2500.0)]
        if args.composition:
            atm_overview_cli.run_atm_overview(_as_atm_overview_args(args, wn_ranges=wn_ranges))
            return

        species = args.species or ["H2O"]
        if len(species) == 1 and len(wn_ranges) == 1:
            molecule_plot_cli.run_overview(_as_molecule_overview_args(args, species=species[0], wn_range=wn_ranges[0]))
        else:
            molecule_plot_cli.run_overview_batch(_as_molecule_overview_batch_args(args, species=species, wn_ranges=wn_ranges))
        return

    parser.error(f"unsupported plot command: {args.command}")
