"""NetCDF dump CLI for spectroscopy data products."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import numpy as np
import xarray as xr

from .atm_overview_cli import compute_mixture_overview_products
from .config import SpectroscopyConfig, parse_broadening_composition, resolve_hitran_cia_pair
from .hitran_cia import load_cia_dataset
from .hitran_lines import build_line_provider, download_hitran_lines
from .output_names import default_output_path as default_named_output_path
from .shared_cli import (
    HelpFormatter,
    build_band,
    default_hitran_dir,
    parse_wn_range,
)
from .spectrum import (
    _resolve_continuum_sources,
    compute_absorption_spectrum_from_sources,
    spectrum_to_dataset,
    write_spectrum_dataset,
)
from .transmittance import compute_transmittance_spectrum, transmittance_to_dataset, write_transmittance_dataset


class _ZeroLineProvider:
    def cross_section_cm2_molecule(self, *, wavenumber_grid_cm1, temperature_k, pressure_pa):
        return np.zeros_like(np.asarray(wavenumber_grid_cm1, dtype=np.float64))


def _add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pair", default=None, metavar="PAIR", help="CIA pair target, for example H2-H2 or H2-He.")
    parser.add_argument("--species", default=None, metavar="NAME", help="Molecular target, for example CO2, H2O, CH4, NH3, or H2S.")
    parser.add_argument("--composition", default=None, metavar="SPECIES:FRACTION,...", help="Gas mixture target, for example H2:0.9,He:0.1,CH4:0.004.")


def _add_common_arguments(parser: argparse.ArgumentParser, *, include_path_length: bool = False) -> None:
    _add_selector_arguments(parser)
    parser.add_argument("--output", type=Path, default=None, metavar="PATH", help="Output NetCDF path. Defaults to an auto-generated path under output/.")
    parser.add_argument("--hitran-dir", type=Path, default=default_hitran_dir(), metavar="DIR", help="Directory for downloaded HITRAN line and CIA data.")
    parser.add_argument("--temperature-k", type=float, default=300.0, metavar="K", help="Gas temperature in kelvin.")
    parser.add_argument("--pressure-bar", type=float, default=1.0, metavar="BAR", help="Gas pressure in bar.")
    parser.add_argument("--wn-range", dest="wn_ranges", action="append", type=parse_wn_range, metavar="MIN,MAX", help="Wavenumber range in cm^-1. Repeat to store multiple bands in one NetCDF.")
    parser.add_argument("--resolution", type=float, default=1.0, metavar="CM^-1", help="Wavenumber grid spacing in cm^-1.")
    parser.add_argument("--refresh-hitran", action="store_true", help="Re-download HITRAN line tables even if cached.")
    parser.add_argument("--broadening-composition", default=None, metavar="BROADENER:FRACTION,...", help="Line-broadening gas composition for molecular line calculations, for example air:0.8,self:0.2 or H2:0.85,He:0.15.")
    parser.add_argument("--filename", default=None, metavar="FILE", help="Use a specific CIA filename instead of resolving one from --pair.")
    parser.add_argument("--cia-filename", default=None, metavar="FILE", help="Optional CIA filename to include for molecular targets.")
    parser.add_argument("--cia-pair", default=None, metavar="PAIR", help="Optional CIA pair to include for molecular targets.")
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/", metavar="URL", help="HITRAN CIA index URL used to resolve CIA files.")
    parser.add_argument("--refresh-cia", action="store_true", help="Re-download HITRAN CIA files even if cached.")
    if include_path_length:
        parser.add_argument("--path-length-m", type=float, default=1.0, metavar="M", help="Propagation path length in meters.")


def _validate_single_selector(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    selectors = [bool(args.pair), bool(args.species), bool(args.composition)]
    if sum(selectors) > 1:
        parser.error("choose only one of --pair, --species, or --composition")


def _selected_target(args: argparse.Namespace) -> tuple[str, object]:
    if args.pair:
        return "pair", args.pair
    if args.composition:
        return "composition", args.composition
    return "species", args.species or "CO2"


def _default_cli_output_path(args: argparse.Namespace, *, suffix: str) -> Path:
    _, target = _selected_target(args)
    return default_named_output_path(
        target_name=target,
        plot_type=args.command,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        wn_range=args.wn_range,
        suffix=suffix,
        output_dir=Path("output"),
    )


def _selected_wn_ranges(args: argparse.Namespace) -> list[tuple[float, float]]:
    return list(args.wn_ranges or [(20.0, 2500.0)])


def _combined_wn_range(wn_ranges: list[tuple[float, float]]) -> tuple[float, float]:
    return min(wn_min for wn_min, _ in wn_ranges), max(wn_max for _, wn_max in wn_ranges)


def _args_for_wn_range(args: argparse.Namespace, wn_range: tuple[float, float]) -> argparse.Namespace:
    scoped = argparse.Namespace(**vars(args))
    scoped.wn_range = wn_range
    return scoped


def _multi_range_output_path(args: argparse.Namespace, *, suffix: str) -> Path:
    if args.output is not None:
        return args.output
    scoped = _args_for_wn_range(args, _combined_wn_range(_selected_wn_ranges(args)))
    return _default_cli_output_path(scoped, suffix=suffix)


def _combine_band_datasets(datasets: list[xr.Dataset], *, wn_ranges: list[tuple[float, float]]) -> xr.Dataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    max_points = max(int(dataset.sizes["wavenumber_cm1"]) for dataset in datasets)
    band_count = len(datasets)
    coords: dict[str, object] = {
        "band": ("band", np.arange(band_count, dtype=np.int64)),
        "wavenumber_index": ("wavenumber_index", np.arange(max_points, dtype=np.int64)),
    }
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    wavenumber_values = np.full((band_count, max_points), np.nan, dtype=np.float64)
    band_sizes = np.zeros(band_count, dtype=np.int64)
    band_min = np.zeros(band_count, dtype=np.float64)
    band_max = np.zeros(band_count, dtype=np.float64)

    variable_names = tuple(datasets[0].data_vars)
    for name in variable_names:
        data_vars[name] = (("band", "wavenumber_index"), np.full((band_count, max_points), np.nan, dtype=np.float64))

    for band_index, (dataset, wn_range) in enumerate(zip(datasets, wn_ranges, strict=True)):
        size = int(dataset.sizes["wavenumber_cm1"])
        band_sizes[band_index] = size
        band_min[band_index], band_max[band_index] = wn_range
        wavenumber_values[band_index, :size] = np.asarray(dataset["wavenumber_cm1"].values, dtype=np.float64)
        for name in variable_names:
            data_vars[name][1][band_index, :size] = np.asarray(dataset[name].values, dtype=np.float64)

    data_vars["wavenumber_cm1"] = (("band", "wavenumber_index"), wavenumber_values)
    data_vars["band_size"] = (("band",), band_sizes)
    data_vars["band_wavenumber_min_cm1"] = (("band",), band_min)
    data_vars["band_wavenumber_max_cm1"] = (("band",), band_max)

    combined = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(datasets[0].attrs))
    combined.attrs["num_bands"] = band_count
    return combined


def _write_combined_dataset(datasets: list[xr.Dataset], *, wn_ranges: list[tuple[float, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = _combine_band_datasets(datasets, wn_ranges=wn_ranges)
    combined.to_netcdf(output_path)
    combined.close()
    for dataset in datasets:
        dataset.close()


def _xsection_dataset(spectrum) -> xr.Dataset:
    dataset = spectrum_to_dataset(spectrum)
    keep = ("sigma_line_cm2_molecule", "sigma_cia_cm2_molecule", "sigma_total_cm2_molecule")
    drop = [name for name in dataset.data_vars if name not in keep]
    if drop:
        dataset = dataset.drop_vars(drop)
    return dataset


def _write_xsection_dataset(spectrum, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = _xsection_dataset(spectrum)
    dataset.to_netcdf(output_path)
    dataset.close()


def _resolve_pair_filename(args: argparse.Namespace) -> tuple[str, str]:
    pair = str(args.pair or "H2-H2")
    if args.filename:
        return pair, str(args.filename)
    metadata = resolve_hitran_cia_pair(pair)
    return metadata.pair, metadata.filename


def _resolve_species_cia(args: argparse.Namespace, config: SpectroscopyConfig):
    explicit_filename = getattr(args, "cia_filename", None)
    explicit_pair = getattr(args, "cia_pair", None)
    if explicit_filename:
        pair = str(explicit_pair or f"{config.hitran_species.name}-{config.hitran_species.name}")
        filename = str(explicit_filename)
    elif explicit_pair:
        metadata = resolve_hitran_cia_pair(str(explicit_pair))
        pair = metadata.pair
        filename = metadata.filename
    elif config.hitran_species.cia_filename is not None:
        pair = config.cia_pair
        filename = config.cia_filename
    else:
        return None
    return load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=filename,
        pair=pair,
        index_url=str(args.cia_index_url),
        refresh=bool(args.refresh_cia),
    )


def _compute_species_xsection(args: argparse.Namespace):
    species_name = args.species or "CO2"
    band = build_band(args)
    config = SpectroscopyConfig(
        output_path=Path("output") / "unused.nc",
        hitran_cache_dir=args.hitran_dir,
        species_name=species_name,
        broadening_composition=parse_broadening_composition(args.broadening_composition),
        refresh_hitran=args.refresh_hitran,
    )
    line_db = download_hitran_lines(config, band)
    line_provider = build_line_provider(config, line_db)
    cia_dataset = _resolve_species_cia(args, config)
    temperature_k = float(args.temperature_k)
    pressure_pa = float(args.pressure_bar) * 1.0e5
    grid = band.grid()
    cia_cross_section_cm2_molecule = None
    if cia_dataset is None:
        cia_dataset, cia_cross_section_cm2_molecule = _resolve_continuum_sources(
            config=config,
            wavenumber_grid_cm1=grid,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
        )
    spectrum = compute_absorption_spectrum_from_sources(
        species_name=config.hitran_species.name,
        wavenumber_grid_cm1=grid,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        line_provider=line_provider,
        cia_dataset=cia_dataset,
        cia_cross_section_cm2_molecule=cia_cross_section_cm2_molecule,
    )
    return spectrum, line_provider.broadening_summary()


def _compute_pair_xsection(args: argparse.Namespace):
    pair, filename = _resolve_pair_filename(args)
    band = build_band(args)
    cia_dataset = load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=filename,
        pair=pair,
        index_url=str(args.cia_index_url),
        refresh=bool(args.refresh_cia),
    )
    spectrum = compute_absorption_spectrum_from_sources(
        species_name=pair,
        wavenumber_grid_cm1=band.grid(),
        temperature_k=float(args.temperature_k),
        pressure_pa=float(args.pressure_bar) * 1.0e5,
        line_provider=_ZeroLineProvider(),
        cia_dataset=cia_dataset,
    )
    return spectrum


def _compute_composition_products(args: argparse.Namespace):
    mixture_args = argparse.Namespace(
        composition=args.composition,
        hitran_dir=args.hitran_dir,
        temperature_k=args.temperature_k,
        pressure_bar=args.pressure_bar,
        resolution=args.resolution,
        cia_index_url=args.cia_index_url,
        refresh_hitran=args.refresh_hitran,
        refresh_cia=args.refresh_cia,
        broadening_composition=args.broadening_composition,
        path_length_km=getattr(args, "path_length_m", 1000.0) / 1000.0,
        manifest=None,
    )
    return compute_mixture_overview_products(mixture_args, wn_range=args.wn_range)


def build_parser() -> argparse.ArgumentParser:
    """Create the dump CLI parser."""
    parser = argparse.ArgumentParser(
        description="Write spectroscopy NetCDF products from HITRAN line and CIA data.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump xsection --pair H2-He --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump transmission --composition H2:0.9,He:0.1,CH4:0.004 --path-length-m 1000 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

            Run "pyharp-dump COMMAND -h" for command-specific options.
            """
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    xsection = subparsers.add_parser(
        "xsection",
        help="Write an absorption cross-section dataset.",
        description="Compute line, CIA, and total absorption cross sections and write a NetCDF dataset.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump xsection --species H2O --cia-pair H2O-H2O --wn-range=1000,1500
              pyharp-dump xsection --pair H2-He --temperature-k 500 --wn-range=20,10000
              pyharp-dump xsection --composition H2:0.9,He:0.1,CH4:0.004 --broadening-composition H2:0.9,He:0.1 --wn-range=20,2500
            """
        ),
    )
    _add_common_arguments(xsection)

    transmission = subparsers.add_parser(
        "transmission",
        help="Write a transmission dataset.",
        description="Compute line, CIA, and total transmission over a fixed path length and write a NetCDF dataset.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump transmission --species CO2 --path-length-m 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump transmission --pair H2-H2 --path-length-m 1000 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
              pyharp-dump transmission --composition H2:0.9,He:0.1,CH4:0.004 --path-length-m 1000 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
            """
        ),
    )
    _add_common_arguments(transmission, include_path_length=True)
    return parser


def main() -> None:
    """Run the dump CLI."""
    parser = build_parser()
    args = parser.parse_args()
    _validate_single_selector(args, parser)
    if args.command == "xsection":
        wn_ranges = _selected_wn_ranges(args)
        target_kind, _ = _selected_target(args)
        datasets: list[xr.Dataset] = []
        for wn_range in wn_ranges:
            range_args = _args_for_wn_range(args, wn_range)
            broadening_summary: str | None = None
            if target_kind == "pair":
                spectrum = _compute_pair_xsection(range_args)
            elif target_kind == "composition":
                products = _compute_composition_products(range_args)
                spectrum = products.spectrum
            else:
                spectrum, broadening_summary = _compute_species_xsection(range_args)
            if len(wn_ranges) == 1:
                output_path = _multi_range_output_path(args, suffix=".nc")
                _write_xsection_dataset(spectrum, output_path)
                print(f"Wrote NetCDF: {output_path}")
            else:
                datasets.append(_xsection_dataset(spectrum))
            if broadening_summary:
                print(f"Broadening: {broadening_summary}")
        if len(wn_ranges) > 1:
            output_path = _multi_range_output_path(args, suffix=".nc")
            _write_combined_dataset(datasets, wn_ranges=wn_ranges, output_path=output_path)
            print(f"Wrote NetCDF: {output_path}")
        return
    if args.command == "transmission":
        wn_ranges = _selected_wn_ranges(args)
        target_kind, _ = _selected_target(args)
        datasets: list[xr.Dataset] = []
        for wn_range in wn_ranges:
            range_args = _args_for_wn_range(args, wn_range)
            broadening_summary: str | None = None
            if target_kind == "pair":
                spectrum = _compute_pair_xsection(range_args)
                transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=range_args.path_length_m)
            elif target_kind == "composition":
                products = _compute_composition_products(range_args)
                transmittance = products.transmittance
            else:
                spectrum, broadening_summary = _compute_species_xsection(range_args)
                transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=range_args.path_length_m)
            if len(wn_ranges) == 1:
                output_path = _multi_range_output_path(args, suffix=".nc")
                write_transmittance_dataset(transmittance, output_path)
                print(f"Wrote NetCDF: {output_path}")
            else:
                datasets.append(transmittance_to_dataset(transmittance))
            if broadening_summary:
                print(f"Broadening: {broadening_summary}")
        if len(wn_ranges) > 1:
            output_path = _multi_range_output_path(args, suffix=".nc")
            _write_combined_dataset(datasets, wn_ranges=wn_ranges, output_path=output_path)
            print(f"Wrote NetCDF: {output_path}")
        return
