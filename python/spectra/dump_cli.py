"""NetCDF dump CLI for spectroscopy data products."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
from textwrap import dedent

import numpy as np
import xarray as xr

from .atm_overview_cli import _parse_composition, compute_mixture_overview_products
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
    number_density_cm3_from_pressure_temperature,
    spectrum_to_dataset,
)
from .transmittance import compute_transmittance_spectrum


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


def _clean_var_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")
    return "_".join(part for part in token.split("_") if part)


def _combine_band_datasets(datasets: list[xr.Dataset], *, wn_ranges: list[tuple[float, float]]) -> xr.Dataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    max_points = max(int(dataset.sizes["wavenumber"]) for dataset in datasets)
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
        size = int(dataset.sizes["wavenumber"])
        band_sizes[band_index] = size
        band_min[band_index], band_max[band_index] = wn_range
        wavenumber_values[band_index, :size] = np.asarray(dataset["wavenumber"].values, dtype=np.float64)
        for name in variable_names:
            data_vars[name][1][band_index, :size] = np.asarray(dataset[name].values, dtype=np.float64)

    data_vars["wavenumber"] = (("band", "wavenumber_index"), wavenumber_values)
    data_vars["band_size"] = (("band",), band_sizes)
    data_vars["band_wavenumber_min"] = (("band",), band_min)
    data_vars["band_wavenumber_max"] = (("band",), band_max)

    combined = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(datasets[0].attrs))
    combined.attrs["num_bands"] = band_count
    for name in variable_names:
        combined[name].attrs = dict(datasets[0][name].attrs)
    if "wavenumber" in datasets[0]:
        combined["wavenumber"].attrs = dict(datasets[0]["wavenumber"].attrs)
    combined["band_size"].attrs = {"long_name": "band sample count", "units": "1"}
    combined["band_wavenumber_min"].attrs = {"long_name": "band minimum wavenumber", "units": "cm^-1"}
    combined["band_wavenumber_max"].attrs = {"long_name": "band maximum wavenumber", "units": "cm^-1"}
    return combined


def _write_combined_dataset(datasets: list[xr.Dataset], *, wn_ranges: list[tuple[float, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = _combine_band_datasets(datasets, wn_ranges=wn_ranges)
    try:
        _write_dataset_via_tmp(combined, output_path, engine="scipy")
    finally:
        combined.close()
        for dataset in datasets:
            dataset.close()


def _xsection_dataset(
    spectrum,
    *,
    species_name: str | None = None,
    secondary_component: dict[str, object] | None = None,
) -> xr.Dataset:
    dataset = spectrum_to_dataset(spectrum)
    keep = ("sigma_line_cm2_molecule", "sigma_cia_cm2_molecule", "sigma_total_cm2_molecule")
    drop = [name for name in dataset.data_vars if name not in keep]
    if drop:
        dataset = dataset.drop_vars(drop)
    species_token = _clean_var_token(species_name or spectrum.species_name)
    rename_map = {
        "wavenumber_cm1": "wavenumber",
        "sigma_line_cm2_molecule": f"sigma_line_{species_token}",
        "sigma_total_cm2_molecule": "sigma_total",
    }
    keep_cia_generic = True
    if secondary_component is not None:
        kind = str(secondary_component.get("kind", ""))
        label = _clean_var_token(str(secondary_component.get("label", "")))
        if kind == "continuum" and label:
            rename_map["sigma_cia_cm2_molecule"] = f"sigma_continuum_{label}"
            keep_cia_generic = False
        elif kind in {"self_cia", "binary_cia"} and label:
            rename_map["sigma_cia_cm2_molecule"] = f"sigma_cia_{label}"
            keep_cia_generic = False
    if keep_cia_generic:
        rename_map["sigma_cia_cm2_molecule"] = "sigma_cia"
    dataset = dataset.rename_vars(rename_map)
    dataset = dataset.swap_dims({"wavenumber_cm1": "wavenumber"})
    dataset["wavenumber"].attrs = {"long_name": "wavenumber", "units": "cm^-1"}
    dataset[f"sigma_line_{species_token}"].attrs = {"long_name": f"{species_name or spectrum.species_name} line absorption cross section", "units": "cm^2 molecule^-1"}
    dataset["sigma_total"].attrs = {"long_name": "total absorption cross section", "units": "cm^2 molecule^-1"}
    cia_name = rename_map["sigma_cia_cm2_molecule"]
    if cia_name == "sigma_cia":
        dataset["sigma_cia"].attrs = {"long_name": "CIA or continuum absorption cross section", "units": "cm^2 molecule^-1"}
    elif cia_name.startswith("sigma_continuum_"):
        dataset[cia_name].attrs = {"long_name": f"{secondary_component['label']} absorption cross section", "units": "cm^2 molecule^-1"}
    else:
        dataset[cia_name].attrs = {"long_name": f"{secondary_component['label']} CIA absorption cross section", "units": "cm^2 molecule^-1"}
        binary = np.asarray(secondary_component["binary_absorption_coefficient"], dtype=np.float64)
        dataset[f"binary_absorption_coefficient_{_clean_var_token(str(secondary_component['label']))}"] = (
            "wavenumber",
            binary,
            {"long_name": f"{secondary_component['label']} CIA binary absorption coefficient", "units": "cm^5 molecule^-2"},
        )
    return dataset


def _composition_xsection_dataset(args: argparse.Namespace) -> xr.Dataset:
    products = _compute_composition_products(args)
    spectrum = products.spectrum
    number_density_cm3 = float(
        getattr(
            spectrum,
            "number_density_cm3",
            number_density_cm3_from_pressure_temperature(
                pressure_pa=float(spectrum.pressure_pa),
                temperature_k=float(spectrum.temperature_k),
            ),
        )
    )
    data_vars: dict[str, tuple[tuple[str], np.ndarray, dict[str, str]]] = {
        "wavenumber": ("wavenumber", np.asarray(spectrum.wavenumber_cm1, dtype=np.float64), {"long_name": "wavenumber", "units": "cm^-1"}),
        "sigma_total": (
            "wavenumber",
            np.asarray(spectrum.sigma_total_cm2_molecule, dtype=np.float64),
            {"long_name": "total absorption cross section", "units": "cm^2 molecule^-1"},
        ),
    }
    for term in products.species_terms:
        name = f"sigma_line_{_clean_var_token(term.species_name)}"
        data_vars[name] = (
            "wavenumber",
            np.asarray(term.sigma_line_cm2_molecule, dtype=np.float64),
            {"long_name": f"{term.species_name} line absorption cross section", "units": "cm^2 molecule^-1"},
        )
    for source in products.secondary_sources:
        label_token = _clean_var_token(source.label)
        if source.kind == "continuum":
            name = f"sigma_continuum_{label_token}"
            if source.weight > 0.0:
                values = np.asarray(source.sigma_cm2_molecule, dtype=np.float64) / float(source.weight)
            else:
                values = np.zeros_like(np.asarray(source.sigma_cm2_molecule, dtype=np.float64))
            attrs = {"long_name": f"{source.label} absorption cross section", "units": "cm^2 molecule^-1"}
        elif source.kind in {"self_cia", "binary_cia"}:
            if source.weight > 0.0 and number_density_cm3 > 0.0:
                values = np.asarray(source.sigma_cm2_molecule, dtype=np.float64) / (float(source.weight) * number_density_cm3)
            else:
                values = np.zeros_like(np.asarray(source.sigma_cm2_molecule, dtype=np.float64))
            name = f"binary_absorption_coefficient_{label_token}"
            attrs = {"long_name": f"{source.label} CIA binary absorption coefficient", "units": "cm^5 molecule^-2"}
        else:
            continue
        data_vars[name] = ("wavenumber", values, attrs)
    dataset = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.asarray(spectrum.wavenumber_cm1, dtype=np.float64))},
        data_vars={name: value for name, value in data_vars.items() if name != "wavenumber"},
        attrs={
            "composition_input": str(args.composition),
            "species_name": ",".join(_composition_species_names(str(args.composition))),
            "temperature_k": spectrum.temperature_k,
            "pressure_pa": spectrum.pressure_pa,
            "pressure_bar": spectrum.pressure_pa / 1.0e5,
        },
    )
    dataset["wavenumber"].attrs = {"long_name": "wavenumber", "units": "cm^-1"}
    return dataset


def _species_transmission_dataset(
    *,
    spectrum,
    transmittance,
    species_name: str,
    secondary_component: dict[str, object] | None = None,
) -> xr.Dataset:
    species_token = _clean_var_token(species_name)
    data_vars: dict[str, tuple[tuple[str], np.ndarray, dict[str, str]]] = {
        f"transmittance_line_{species_token}": (
            "wavenumber",
            np.asarray(transmittance.transmittance_line, dtype=np.float64),
            {"long_name": f"{species_name} line transmittance", "units": "1"},
        ),
        f"attenuation_line_{species_token}": (
            "wavenumber",
            np.asarray(spectrum.attenuation_line_m1, dtype=np.float64),
            {"long_name": f"{species_name} line attenuation coefficient", "units": "m^-1"},
        ),
        "transmittance_total": (
            "wavenumber",
            np.asarray(transmittance.transmittance_total, dtype=np.float64),
            {"long_name": "total transmittance", "units": "1"},
        ),
        "attenuation_total": (
            "wavenumber",
            np.asarray(spectrum.attenuation_total_m1, dtype=np.float64),
            {"long_name": "total attenuation coefficient", "units": "m^-1"},
        ),
    }
    if secondary_component is None:
        data_vars["transmittance_cia"] = (
            "wavenumber",
            np.asarray(transmittance.transmittance_cia, dtype=np.float64),
            {"long_name": "CIA or continuum transmittance", "units": "1"},
        )
        data_vars["attenuation_cia"] = (
            "wavenumber",
            np.asarray(spectrum.attenuation_cia_m1, dtype=np.float64),
            {"long_name": "CIA or continuum attenuation coefficient", "units": "m^-1"},
        )
    else:
        label = str(secondary_component.get("label", ""))
        label_token = _clean_var_token(label)
        if str(secondary_component.get("kind", "")) == "continuum":
            trans_name = f"transmittance_continuum_{label_token}"
            att_name = f"attenuation_continuum_{label_token}"
            trans_long = f"{label} transmittance"
            att_long = f"{label} attenuation coefficient"
        else:
            trans_name = f"transmittance_cia_{label_token}"
            att_name = f"attenuation_cia_{label_token}"
            trans_long = f"{label} CIA transmittance"
            att_long = f"{label} CIA attenuation coefficient"
        data_vars[trans_name] = (
            "wavenumber",
            np.asarray(transmittance.transmittance_cia, dtype=np.float64),
            {"long_name": trans_long, "units": "1"},
        )
        data_vars[att_name] = (
            "wavenumber",
            np.asarray(spectrum.attenuation_cia_m1, dtype=np.float64),
            {"long_name": att_long, "units": "m^-1"},
        )
    dataset = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.asarray(transmittance.wavenumber_cm1, dtype=np.float64))},
        data_vars=data_vars,
        attrs={
            "species_name": str(species_name),
            "path_length_m": float(transmittance.path_length_m),
            "temperature_k": float(transmittance.temperature_k),
            "pressure_pa": float(transmittance.pressure_pa),
            "pressure_bar": float(transmittance.pressure_pa) / 1.0e5,
        },
    )
    dataset["wavenumber"].attrs = {"long_name": "wavenumber", "units": "cm^-1"}
    return dataset


def _pair_transmission_dataset(args: argparse.Namespace) -> xr.Dataset:
    spectrum = _compute_pair_xsection(args)
    transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=args.path_length_m)
    pair, _ = _resolve_pair_filename(args)
    pair_token = _clean_var_token(pair)
    dataset = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.asarray(transmittance.wavenumber_cm1, dtype=np.float64))},
        data_vars={
            f"transmittance_cia_{pair_token}": (
                "wavenumber",
                np.asarray(transmittance.transmittance_cia, dtype=np.float64),
                {"long_name": f"{pair} CIA transmittance", "units": "1"},
            ),
            f"attenuation_cia_{pair_token}": (
                "wavenumber",
                np.asarray(spectrum.attenuation_cia_m1, dtype=np.float64),
                {"long_name": f"{pair} CIA attenuation coefficient", "units": "m^-1"},
            ),
            "transmittance_total": (
                "wavenumber",
                np.asarray(transmittance.transmittance_total, dtype=np.float64),
                {"long_name": "total transmittance", "units": "1"},
            ),
            "attenuation_total": (
                "wavenumber",
                np.asarray(spectrum.attenuation_total_m1, dtype=np.float64),
                {"long_name": "total attenuation coefficient", "units": "m^-1"},
            ),
        },
        attrs={
            "species_name": pair,
            "path_length_m": float(transmittance.path_length_m),
            "temperature_k": float(transmittance.temperature_k),
            "pressure_pa": float(transmittance.pressure_pa),
            "pressure_bar": float(transmittance.pressure_pa) / 1.0e5,
        },
    )
    dataset["wavenumber"].attrs = {"long_name": "wavenumber", "units": "cm^-1"}
    return dataset


def _composition_transmission_dataset(args: argparse.Namespace) -> xr.Dataset:
    products = _compute_composition_products(args)
    spectrum = products.spectrum
    transmittance = products.transmittance
    number_density_cm3 = float(
        getattr(
            spectrum,
            "number_density_cm3",
            number_density_cm3_from_pressure_temperature(
                pressure_pa=float(spectrum.pressure_pa),
                temperature_k=float(spectrum.temperature_k),
            ),
        )
    )
    path_length_m = float(transmittance.path_length_m)
    data_vars: dict[str, tuple[tuple[str], np.ndarray, dict[str, str]]] = {
        "transmittance_total": (
            "wavenumber",
            np.asarray(transmittance.transmittance_total, dtype=np.float64),
            {"long_name": "total transmittance", "units": "1"},
        ),
        "attenuation_total": (
            "wavenumber",
            np.asarray(spectrum.attenuation_total_m1, dtype=np.float64),
            {"long_name": "total attenuation coefficient", "units": "m^-1"},
        ),
    }
    for term in products.species_terms:
        species_token = _clean_var_token(term.species_name)
        attenuation = np.asarray(term.mole_fraction * term.sigma_line_cm2_molecule * number_density_cm3 * 100.0, dtype=np.float64)
        data_vars[f"attenuation_line_{species_token}"] = (
            "wavenumber",
            attenuation,
            {"long_name": f"{term.species_name} weighted line attenuation coefficient", "units": "m^-1"},
        )
        data_vars[f"transmittance_line_{species_token}"] = (
            "wavenumber",
            np.exp(-attenuation * path_length_m),
            {"long_name": f"{term.species_name} weighted line transmittance", "units": "1"},
        )
    for source in products.secondary_sources:
        label_token = _clean_var_token(source.label)
        attenuation = np.asarray(source.sigma_cm2_molecule * number_density_cm3 * 100.0, dtype=np.float64)
        if source.kind == "continuum":
            att_name = f"attenuation_continuum_{label_token}"
            trans_name = f"transmittance_continuum_{label_token}"
            prefix = f"{source.label} weighted"
        else:
            att_name = f"attenuation_cia_{label_token}"
            trans_name = f"transmittance_cia_{label_token}"
            prefix = f"{source.label} weighted CIA"
        data_vars[att_name] = (
            "wavenumber",
            attenuation,
            {"long_name": f"{prefix} attenuation coefficient", "units": "m^-1"},
        )
        data_vars[trans_name] = (
            "wavenumber",
            np.exp(-attenuation * path_length_m),
            {"long_name": f"{prefix} transmittance", "units": "1"},
        )
    dataset = xr.Dataset(
        coords={"wavenumber": ("wavenumber", np.asarray(transmittance.wavenumber_cm1, dtype=np.float64))},
        data_vars=data_vars,
        attrs={
            "composition_input": str(args.composition),
            "species_name": ",".join(_composition_species_names(str(args.composition))),
            "path_length_m": path_length_m,
            "temperature_k": float(transmittance.temperature_k),
            "pressure_pa": float(transmittance.pressure_pa),
            "pressure_bar": float(transmittance.pressure_pa) / 1.0e5,
        },
    )
    dataset["wavenumber"].attrs = {"long_name": "wavenumber", "units": "cm^-1"}
    return dataset


def _write_xsection_dataset(
    spectrum,
    output_path: Path,
    *,
    species_name: str | None = None,
    secondary_component: dict[str, object] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = _xsection_dataset(spectrum, species_name=species_name, secondary_component=secondary_component)
    try:
        _write_dataset_via_tmp(dataset, output_path, engine="scipy")
    finally:
        dataset.close()


def _write_dataset_via_tmp(dataset: xr.Dataset, output_path: Path, *, engine: str) -> None:
    """Write to a local temporary file first, then move to the target path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="pyharp_", suffix=".nc", dir="/tmp")
    import os

    os.close(fd)
    try:
        Path(tmp_name).unlink(missing_ok=True)
        dataset.to_netcdf(tmp_name, engine=engine)
        shutil.move(tmp_name, output_path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _resolve_pair_filename(args: argparse.Namespace) -> tuple[str, str]:
    pair = str(args.pair or "H2-H2")
    if args.filename:
        return pair, str(args.filename)
    metadata = resolve_hitran_cia_pair(pair)
    return metadata.pair, metadata.filename


def _resolve_species_cia_selection(args: argparse.Namespace, config: SpectroscopyConfig) -> tuple[str, str] | None:
    explicit_filename = getattr(args, "cia_filename", None)
    explicit_pair = getattr(args, "cia_pair", None)
    if explicit_filename:
        return str(explicit_pair or f"{config.hitran_species.name}-{config.hitran_species.name}"), str(explicit_filename)
    if explicit_pair:
        metadata = resolve_hitran_cia_pair(str(explicit_pair))
        return metadata.pair, metadata.filename
    if config.hitran_species.cia_filename is not None:
        return config.cia_pair, config.cia_filename
    return None


def _resolve_species_cia(args: argparse.Namespace, config: SpectroscopyConfig):
    selection = _resolve_species_cia_selection(args, config)
    if selection is None:
        return None
    pair, filename = selection
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
    cia_selection = _resolve_species_cia_selection(args, config)
    temperature_k = float(args.temperature_k)
    pressure_pa = float(args.pressure_bar) * 1.0e5
    grid = band.grid()
    cia_cross_section_cm2_molecule = None
    secondary_component: dict[str, object] | None = None
    if cia_dataset is None:
        cia_dataset, cia_cross_section_cm2_molecule = _resolve_continuum_sources(
            config=config,
            wavenumber_grid_cm1=grid,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
        )
        if cia_cross_section_cm2_molecule is not None:
            secondary_component = {"kind": "continuum", "label": "H2O continuum (MT_CKD)"}
    elif cia_selection is not None:
        pair, _ = cia_selection
        secondary_component = {
            "kind": "binary_cia",
            "label": pair,
            "binary_absorption_coefficient": np.asarray(
                cia_dataset.interpolate_to_grid(
                    temperature_k=temperature_k,
                    wavenumber_grid_cm1=grid,
                ),
                dtype=np.float64,
            ),
        }
    spectrum = compute_absorption_spectrum_from_sources(
        species_name=config.hitran_species.name,
        wavenumber_grid_cm1=grid,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        line_provider=line_provider,
        cia_dataset=cia_dataset,
        cia_cross_section_cm2_molecule=cia_cross_section_cm2_molecule,
    )
    return spectrum, line_provider.broadening_summary(), secondary_component


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


def _pair_xsection_dataset(args: argparse.Namespace) -> xr.Dataset:
    pair, filename = _resolve_pair_filename(args)
    band = build_band(args)
    cia_dataset = load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=filename,
        pair=pair,
        index_url=str(args.cia_index_url),
        refresh=bool(args.refresh_cia),
    )
    grid = band.grid()
    binary = np.asarray(
        cia_dataset.interpolate_to_grid(
            temperature_k=float(args.temperature_k),
            wavenumber_grid_cm1=grid,
        ),
        dtype=np.float64,
    )
    return xr.Dataset(
        coords={
            "wavenumber": ("wavenumber", grid, {"long_name": "wavenumber", "units": "cm^-1"}),
        },
        data_vars={
            "binary_absorption_coefficient": (
                "wavenumber",
                binary,
                {"long_name": f"{pair} CIA binary absorption coefficient", "units": "cm^5 molecule^-2"},
            ),
        },
        attrs={
            "pair_name": pair,
            "temperature_k": float(args.temperature_k),
            "source_filename": filename,
        },
    )


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


def _compute_xsection_band(task: tuple[str, argparse.Namespace]) -> tuple[xr.Dataset, str | None]:
    target_kind, args = task
    broadening_summary: str | None = None
    if target_kind == "pair":
        dataset = _pair_xsection_dataset(args)
        return dataset, broadening_summary
    elif target_kind == "composition":
        dataset = _composition_xsection_dataset(args)
        return dataset, broadening_summary
    else:
        spectrum, broadening_summary, secondary_component = _compute_species_xsection(args)
        dataset = _xsection_dataset(
            spectrum,
            species_name=str(args.species or "CO2"),
            secondary_component=secondary_component,
        )
        return dataset, broadening_summary


def _compute_transmission_band(task: tuple[str, argparse.Namespace]) -> tuple[xr.Dataset, str | None]:
    target_kind, args = task
    broadening_summary: str | None = None
    if target_kind == "pair":
        dataset = _pair_transmission_dataset(args)
    elif target_kind == "composition":
        dataset = _composition_transmission_dataset(args)
    else:
        spectrum, broadening_summary, secondary_component = _compute_species_xsection(args)
        transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=args.path_length_m)
        dataset = _species_transmission_dataset(
            spectrum=spectrum,
            transmittance=transmittance,
            species_name=str(args.species or "CO2"),
            secondary_component=secondary_component,
        )
    return dataset, broadening_summary


def _composition_species_names(composition: str) -> tuple[str, ...]:
    return tuple(_parse_composition(composition).keys())


def _parallel_band_results(
    tasks: list[tuple[str, argparse.Namespace]],
    *,
    worker,
) -> list[tuple[xr.Dataset, str | None]]:
    if len(tasks) <= 1:
        return [worker(task) for task in tasks]
    max_workers = min(len(tasks), os.cpu_count() or 1)
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        return list(executor.map(worker, tasks))


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
        if target_kind in {"pair", "composition"}:
            output_path = _multi_range_output_path(args, suffix=".nc")
            if len(wn_ranges) == 1:
                dataset, _ = _compute_xsection_band((target_kind, _args_for_wn_range(args, wn_ranges[0])))
                try:
                    _write_dataset_via_tmp(dataset, output_path, engine="scipy")
                finally:
                    dataset.close()
                print(f"Wrote NetCDF: {output_path}")
                return
            tasks = [(target_kind, _args_for_wn_range(args, wn_range)) for wn_range in wn_ranges]
            results = _parallel_band_results(tasks, worker=_compute_xsection_band)
            datasets = [dataset for dataset, _ in results]
            _write_combined_dataset(datasets, wn_ranges=wn_ranges, output_path=output_path)
            print(f"Wrote NetCDF: {output_path}")
            return
        if len(wn_ranges) == 1:
            range_args = _args_for_wn_range(args, wn_ranges[0])
            broadening_summary: str | None = None
            spectrum, broadening_summary, secondary_component = _compute_species_xsection(range_args)
            output_path = _multi_range_output_path(args, suffix=".nc")
            _write_xsection_dataset(
                spectrum,
                output_path,
                species_name=str(range_args.species or "CO2"),
                secondary_component=secondary_component,
            )
            print(f"Wrote NetCDF: {output_path}")
            if broadening_summary:
                print(f"Broadening: {broadening_summary}")
            return
        tasks = [("species", _args_for_wn_range(args, wn_range)) for wn_range in wn_ranges]
        results = _parallel_band_results(tasks, worker=_compute_xsection_band)
        datasets = [dataset for dataset, _ in results]
        for _, broadening_summary in results:
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
        if len(wn_ranges) == 1:
            range_args = _args_for_wn_range(args, wn_ranges[0])
            dataset, broadening_summary = _compute_transmission_band((target_kind, range_args))
            output_path = _multi_range_output_path(args, suffix=".nc")
            try:
                _write_dataset_via_tmp(dataset, output_path, engine="scipy")
            finally:
                dataset.close()
            print(f"Wrote NetCDF: {output_path}")
            if broadening_summary:
                print(f"Broadening: {broadening_summary}")
            return
        tasks = [(target_kind, _args_for_wn_range(args, wn_range)) for wn_range in wn_ranges]
        results = _parallel_band_results(tasks, worker=_compute_transmission_band)
        datasets = [dataset for dataset, _ in results]
        for _, broadening_summary in results:
            if broadening_summary:
                print(f"Broadening: {broadening_summary}")
        if len(wn_ranges) > 1:
            output_path = _multi_range_output_path(args, suffix=".nc")
            _write_combined_dataset(datasets, wn_ranges=wn_ranges, output_path=output_path)
            print(f"Wrote NetCDF: {output_path}")
        return
