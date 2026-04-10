"""Unified plotting command for spectroscopy diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import atm_overview_cli, cia_plot_cli, molecule_plot_cli


class _SplitSpeciesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        species: list[str] = []
        for value in values:
            species.extend(part.strip() for part in str(value).split(",") if part.strip())
        setattr(namespace, self.dest, species)


def _add_state_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)


def _add_common_arguments(parser: argparse.ArgumentParser, *, allow_multiple_ranges: bool = False) -> None:
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"))
    parser.add_argument("--resolution", type=float, default=1.0)
    if allow_multiple_ranges:
        parser.add_argument("--wn-range", dest="wn_ranges", action="append", type=molecule_plot_cli._parse_wn_range)
    else:
        parser.add_argument("--wn-range", type=molecule_plot_cli._parse_wn_range, default=None)


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--refresh-hitran", action="store_true")
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/")
    parser.add_argument("--refresh-cia", action="store_true")


def _add_selector_arguments(parser: argparse.ArgumentParser, *, include_composition: bool = False) -> None:
    parser.add_argument("--pair", default=None)
    parser.add_argument("--species", default=None)
    if include_composition:
        parser.add_argument("--composition", default=None)


def _validate_single_selector(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    selectors = [bool(getattr(args, "pair", None)), bool(getattr(args, "species", None)), bool(getattr(args, "composition", None))]
    if sum(selectors) > 1:
        parser.error("choose only one of --pair, --species, or --composition")


def _as_cia_args(args: argparse.Namespace, *, default_pair: str, default_figure: Path) -> argparse.Namespace:
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        filename=getattr(args, "filename", None),
        pair=args.pair or default_pair,
        temperature_k=args.temperature_k,
        wn_range=args.wn_range or (20.0, 10000.0),
        resolution=args.resolution,
        refresh=getattr(args, "refresh", False),
        pressure_bar=getattr(args, "pressure_bar", 1.0),
        path_length_km=getattr(args, "path_length_km", 1.0),
        figure=args.figure or default_figure,
    )


def _as_molecule_args(args: argparse.Namespace, *, default_figure: Path) -> argparse.Namespace:
    return argparse.Namespace(
        hitran_dir=args.hitran_dir,
        species=args.species or "H2O",
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        wn_range=args.wn_range or (20.0, 2500.0),
        resolution=args.resolution,
        refresh_hitran=args.refresh_hitran,
        cia_filename=getattr(args, "cia_filename", None),
        cia_pair=getattr(args, "cia_pair", None),
        cia_index_url=args.cia_index_url,
        refresh_cia=args.refresh_cia,
        path_length_km=getattr(args, "path_length_km", 1.0),
        figure=args.figure or default_figure,
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
        path_length_km=args.path_length_km,
        figure=args.figure or Path("output/molecule_overview_300K_1bar.pdf"),
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
        figure=args.figure or Path("output/molecule_overview_collection.pdf"),
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
        figure=args.figure or Path("output/atm_overview.pdf"),
        manifest=args.manifest,
    )


def _as_atm_args(args: argparse.Namespace, *, default_figure: Path) -> argparse.Namespace:
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
        figure=args.figure or default_figure,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot pyharp spectroscopy diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    binary = subparsers.add_parser("binary", description="Plot a HITRAN CIA binary absorption coefficient spectrum.")
    _add_common_arguments(binary)
    binary.add_argument("--pair", default="H2-H2")
    binary.add_argument("--filename", default=None)
    binary.add_argument("--temperature-k", type=float, default=300.0)
    binary.add_argument("--refresh", action="store_true")
    binary.add_argument("--figure", type=Path, default=None)

    xsection = subparsers.add_parser("xsection", description="Plot molecular absorption cross section.")
    _add_common_arguments(xsection)
    _add_state_arguments(xsection)
    xsection.add_argument("--species", default="H2O")
    _add_cache_arguments(xsection)
    xsection.add_argument("--cia-filename", default=None)
    xsection.add_argument("--cia-pair", default=None)
    xsection.add_argument("--figure", type=Path, default=None)

    for name in ("attenuation", "transmission"):
        subparser = subparsers.add_parser(name, description=f"Plot {name} for a CIA pair or molecule.")
        _add_common_arguments(subparser)
        _add_state_arguments(subparser)
        _add_selector_arguments(subparser, include_composition=True)
        _add_cache_arguments(subparser)
        subparser.add_argument("--filename", default=None)
        subparser.add_argument("--cia-filename", default=None)
        subparser.add_argument("--cia-pair", default=None)
        if name == "transmission":
            subparser.add_argument("--path-length-km", type=float, default=1.0)
        subparser.add_argument("--figure", type=Path, default=None)

    overview = subparsers.add_parser("overview", description="Generate molecule or atmosphere overview plots.")
    _add_common_arguments(overview, allow_multiple_ranges=True)
    _add_state_arguments(overview)
    overview.add_argument("--path-length-km", type=float, default=1.0)
    overview.add_argument("--species", nargs="+", action=_SplitSpeciesAction, default=None)
    overview.add_argument("--composition", default=None)
    _add_cache_arguments(overview)
    overview.add_argument("--cia-filename", default=None)
    overview.add_argument("--cia-pair", default=None)
    overview.add_argument("--figure", type=Path, default=None)
    overview.add_argument("--manifest", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "binary":
        cia_plot_cli.run_binary(_as_cia_args(args, default_pair="H2-H2", default_figure=Path("output/h2_h2_cia_300K.png")))
        return

    if args.command == "xsection":
        molecule_plot_cli.run_xsection(_as_molecule_args(args, default_figure=Path("output/molecule_xsection_300K_1bar.png")))
        return

    if args.command in {"attenuation", "transmission"}:
        _validate_single_selector(args, parser)
        if args.composition:
            default_figure = (
                Path("output/atm_attenuation_300K_1bar.png")
                if args.command == "attenuation"
                else Path("output/atm_transmission_300K_1bar_1km.png")
            )
            atm_args = _as_atm_args(args, default_figure=default_figure)
            wn_range = args.wn_range or (20.0, 2500.0)
            if args.command == "attenuation":
                atm_overview_cli.run_atm_attenuation(atm_args, wn_range=wn_range)
            else:
                atm_overview_cli.run_atm_transmission(atm_args, wn_range=wn_range)
            return
        if args.pair:
            default_figure = (
                Path("output/h2_h2_cia_attenuation_300K_1bar.png")
                if args.command == "attenuation"
                else Path("output/h2_h2_cia_transmission_300K_1bar_1km.png")
            )
            cia_args = _as_cia_args(args, default_pair=args.pair, default_figure=default_figure)
            if args.command == "attenuation":
                cia_plot_cli.run_attenuation(cia_args)
            else:
                cia_plot_cli.run_transmission(cia_args)
            return

        default_figure = (
            Path("output/molecule_attenuation_300K_1bar.png")
            if args.command == "attenuation"
            else Path("output/molecule_transmission_300K_1bar_1km.png")
        )
        molecule_args = _as_molecule_args(args, default_figure=default_figure)
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
