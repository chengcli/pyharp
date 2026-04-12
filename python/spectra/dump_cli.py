"""NetCDF dump CLI for spectroscopy data products."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys
from textwrap import dedent

import numpy as np
import xarray as xr

from .atm_overview_cli import _parse_composition, compute_mixture_overview_products
from .config import SpectroscopyConfig, parse_broadening_composition, resolve_hitran_cia_filename, resolve_hitran_cia_pair
from .dataset_io import (
    DEFAULT_NETCDF_ENGINE,
    WAVENUMBER_ATTRS,
    add_wavenumber_attrs,
    build_state_attrs,
    clean_var_token,
    write_dataset_via_tmp,
)
from .hitran_cia import load_cia_dataset
from .hitran_lines import build_line_provider, download_hitran_lines
from .orton_xiz_cia import load_orton_xiz_cia_dataset, resolve_orton_xiz_cia_filename
from .output_names import _clean_token, _format_value
from .shared_cli import (
    HelpFormatter,
    build_band,
    default_hitran_dir,
    default_orton_xiz_cia_dir,
    parse_wn_range,
    process_pool_context,
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


TRANSMISSION_PATH_LENGTH_HELP = "Transmission path length in kilometers."
DEL_TEMPERATURE_COORD_ATTRS = {"long_name": "temperature anomaly", "units": "K"}
PRESSURE_COORD_ATTRS = {"long_name": "pressure", "units": "Pa"}
TEMPERATURE_VAR_ATTRS = {"long_name": "base temperature", "units": "K"}
LIST_ARG_OPTIONS = {"--temperature-k", "--pressure-bar", "--del-temperature-k"}


class DumpArgumentParser(argparse.ArgumentParser):
    def _normalize_list_option_tokens(self, args: list[str] | None) -> list[str] | None:
        if args is None:
            args = sys.argv[1:]
        normalized: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token in LIST_ARG_OPTIONS and index + 1 < len(args):
                next_token = args[index + 1]
                if next_token.startswith("-") and "," in next_token:
                    normalized.append(f"{token}={next_token}")
                    index += 2
                    continue
            normalized.append(token)
            index += 1
        return normalized

    def parse_known_args(self, args=None, namespace=None):
        return super().parse_known_args(self._normalize_list_option_tokens(args), namespace)


def _parse_float_list(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("value must be a number or comma-separated list of numbers")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must contain numeric values") from exc


def _add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pair", default=None, metavar="PAIR", help="CIA pair target, for example H2-H2 or H2-He.")
    parser.add_argument("--species", default=None, metavar="NAME", help="Molecular target, for example CO2, H2O, CH4, NH3, or H2S.")
    parser.add_argument("--composition", default=None, metavar="SPECIES:FRACTION,...", help="Gas mixture target, for example H2:0.9,He:0.1,CH4:0.004.")


def _add_common_arguments(parser: argparse.ArgumentParser, *, include_path_length: bool = False) -> None:
    _add_selector_arguments(parser)
    parser.add_argument("--output", type=Path, default=None, metavar="PATH", help="Output NetCDF path. Defaults to an auto-generated path under --output-dir.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), metavar="DIR", help="Directory for auto-generated NetCDF output paths.")
    parser.add_argument("--hitran-dir", type=Path, default=default_hitran_dir(), metavar="DIR", help="Directory for downloaded HITRAN line and CIA data.")
    parser.add_argument("--temperature-k", type=_parse_float_list, default=[300.0], metavar="K[,K...]", help="Base gas temperature in kelvin. Use a comma-separated list paired one-to-one with --pressure-bar.")
    parser.add_argument("--pressure-bar", type=_parse_float_list, default=[1.0], metavar="BAR[,BAR...]", help="Gas pressure in bar. Use a comma-separated list paired one-to-one with --temperature-k.")
    parser.add_argument("--del-temperature-k", type=_parse_float_list, default=[0.0], metavar="K[,K...]", help="Temperature anomalies in kelvin applied to each temperature-pressure pair.")
    parser.add_argument("--wn-range", dest="wn_ranges", action="append", type=parse_wn_range, metavar="MIN,MAX", help="Wavenumber range in cm^-1. Repeat to write one NetCDF per band.")
    parser.add_argument("--resolution", type=float, default=1.0, metavar="CM^-1", help="Wavenumber grid spacing in cm^-1.")
    parser.add_argument("--refresh-hitran", action="store_true", help="Re-download HITRAN line tables even if cached.")
    parser.add_argument("--broadening-composition", default=None, metavar="BROADENER:FRACTION,...", help="Line-broadening gas composition for molecular line calculations, for example air:0.8,self:0.2 or H2:0.85,He:0.15.")
    parser.add_argument("--filename", default=None, metavar="FILE", help="Use a specific CIA filename instead of resolving one from --pair.")
    parser.add_argument("--cia-filename", default=None, metavar="FILE", help="Optional CIA filename to include for molecular targets.")
    parser.add_argument("--cia-pair", default=None, metavar="PAIR", help="Optional CIA pair to include for molecular targets.")
    parser.add_argument("--cia-dir", type=Path, default=None, metavar="DIR", help="Directory for alternate CIA tables. HITRAN defaults to --hitran-dir; Orton/Xiz defaults to orton_xiz_cia/.")
    parser.add_argument("--cia-database", choices=("hitran", "orton_xiz"), default="hitran", help="CIA database backend for --pair targets.")
    parser.add_argument("--cia-model", choices=("auto", "2011", "2018", "xiz", "orton"), default="auto", help="CIA model selector. For HITRAN, use auto/2011/2018. For Orton/Xiz, use auto/xiz/orton.")
    parser.add_argument("--cia-state", choices=("eq", "nm"), default="eq", help="Legacy H2 spin-state table when --cia-database=orton_xiz.")
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/", metavar="URL", help="HITRAN CIA index URL used to resolve CIA files.")
    parser.add_argument("--refresh-cia", action="store_true", help="Re-download HITRAN CIA files even if cached.")
    if include_path_length:
        parser.add_argument("--path-length-km", type=float, default=1.0, metavar="KM", help=TRANSMISSION_PATH_LENGTH_HELP)


def _validate_single_selector(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    selectors = [bool(args.pair), bool(args.species), bool(args.composition)]
    if sum(selectors) > 1:
        parser.error("choose only one of --pair, --species, or --composition")


def _validate_state_grid(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        _state_pairs(args)
    except ValueError as exc:
        parser.error(str(exc))


def _selected_target(args: argparse.Namespace) -> tuple[str, object]:
    if args.pair:
        return "pair", args.pair
    if args.composition:
        return "composition", args.composition
    return "species", args.species or "CO2"


def _selected_base_temperatures(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "temperature_k", [300.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _selected_pressure_bars(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "pressure_bar", [1.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _selected_del_temperatures(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "del_temperature_k", [0.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _state_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    temperatures = _selected_base_temperatures(args)
    pressures = _selected_pressure_bars(args)
    if len(temperatures) != len(pressures):
        raise ValueError("--temperature-k and --pressure-bar must have the same number of values")
    return list(zip(temperatures, pressures, strict=True))


def _single_base_state(args: argparse.Namespace) -> tuple[float, float]:
    temperature_k, pressure_bar = _state_pairs(args)[0]
    return float(temperature_k), float(pressure_bar)


def _state_span_token(min_value: float, max_value: float, *, unit: str) -> str:
    return f"{_format_value(min_value)}_{_format_value(max_value, unit)}"


def _temperature_output_value(args: argparse.Namespace) -> str:
    base_temperatures = _selected_base_temperatures(args)
    del_temperatures = _selected_del_temperatures(args)
    actual_temperatures = [
        base_temperature + del_temperature
        for base_temperature in base_temperatures
        for del_temperature in del_temperatures
    ]
    return _state_span_token(min(actual_temperatures), max(actual_temperatures), unit="K")


def _pressure_output_value(args: argparse.Namespace) -> str:
    pressures = _selected_pressure_bars(args)
    return _state_span_token(min(pressures), max(pressures), unit="bar")


def _default_cli_output_path(args: argparse.Namespace, *, suffix: str) -> Path:
    _, target = _selected_target(args)
    wn_min, wn_max = args.wn_range
    stem = "_".join(
        [
            _clean_token(target),
            _clean_token(args.command),
            _temperature_output_value(args),
            _pressure_output_value(args),
            _format_value(wn_min),
            _format_value(wn_max, "cm1"),
        ]
    )
    return Path(args.output_dir) / f"{stem}{suffix}"


def _selected_wn_ranges(args: argparse.Namespace) -> list[tuple[float, float]]:
    return list(args.wn_ranges or [(20.0, 2500.0)])


def _band_attr_values(wn_range: tuple[float, float]) -> dict[str, float]:
    wn_min, wn_max = wn_range
    return {
        "band_wavenumber_min_cm1": float(wn_min),
        "band_wavenumber_max_cm1": float(wn_max),
    }


def _args_for_wn_range(args: argparse.Namespace, wn_range: tuple[float, float]) -> argparse.Namespace:
    scoped = argparse.Namespace(**vars(args))
    scoped.wn_range = wn_range
    return scoped


def _args_for_state(args: argparse.Namespace, *, base_temperature_k: float, pressure_bar: float, del_temperature_k: float) -> argparse.Namespace:
    scoped = argparse.Namespace(**vars(args))
    scoped.base_temperature_k = float(base_temperature_k)
    scoped.del_temperature_k = float(del_temperature_k)
    scoped.temperature_k = float(base_temperature_k + del_temperature_k)
    scoped.pressure_bar = float(pressure_bar)
    return scoped


def _output_path_for_wn_range(args: argparse.Namespace, *, wn_range: tuple[float, float], suffix: str) -> Path:
    if args.output is None:
        scoped = _args_for_wn_range(args, wn_range)
        return _default_cli_output_path(scoped, suffix=suffix)
    output_path = Path(args.output)
    if len(_selected_wn_ranges(args)) <= 1:
        return output_path
    wn_min, wn_max = wn_range
    return output_path.with_name(
        f"{output_path.stem}_{_format_value(wn_min)}_{_format_value(wn_max)}{output_path.suffix or suffix}"
    )


def _continuum_label_token(label: object) -> str:
    token = clean_var_token(label)
    parts = [part for part in token.split("_") if part and part != "continuum"]
    if parts:
        return "_".join(parts)
    return token


def _xsection_dataset(
    spectrum,
    *,
    species_name: str | None = None,
    secondary_component: dict[str, object] | None = None,
    wn_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    dataset = spectrum_to_dataset(spectrum)
    keep = ("sigma_line_cm2_molecule", "sigma_cia_cm2_molecule", "sigma_total_cm2_molecule")
    drop = [name for name in dataset.data_vars if name not in keep]
    if drop:
        dataset = dataset.drop_vars(drop)
    species_token = clean_var_token(species_name or spectrum.species_name)
    rename_map = {
        "wavenumber_cm1": "wavenumber",
        "sigma_line_cm2_molecule": f"sigma_line_{species_token}",
        "sigma_total_cm2_molecule": "sigma_total",
    }
    keep_cia_generic = True
    if secondary_component is not None:
        kind = str(secondary_component.get("kind", ""))
        if kind == "continuum":
            label = _continuum_label_token(str(secondary_component.get("label", "")))
        else:
            label = clean_var_token(str(secondary_component.get("label", "")))
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
    add_wavenumber_attrs(dataset)
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
        dataset[f"binary_absorption_coefficient_{clean_var_token(str(secondary_component['label']))}"] = (
            "wavenumber",
            binary,
            {"long_name": f"{secondary_component['label']} CIA binary absorption coefficient", "units": "cm^5 molecule^-2"},
        )
    if wn_range is not None:
        dataset.attrs.update(_band_attr_values(wn_range))
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
        name = f"sigma_line_{clean_var_token(term.species_name)}"
        data_vars[name] = (
            "wavenumber",
            np.asarray(term.sigma_line_cm2_molecule, dtype=np.float64),
            {"long_name": f"{term.species_name} line absorption cross section", "units": "cm^2 molecule^-1"},
        )
    for source in products.secondary_sources:
        if source.kind == "continuum":
            label_token = _continuum_label_token(source.label)
            name = f"sigma_continuum_{label_token}"
            if source.weight > 0.0:
                values = np.asarray(source.sigma_cm2_molecule, dtype=np.float64) / float(source.weight)
            else:
                values = np.zeros_like(np.asarray(source.sigma_cm2_molecule, dtype=np.float64))
            attrs = {"long_name": f"{source.label} absorption cross section", "units": "cm^2 molecule^-1"}
        elif source.kind in {"self_cia", "binary_cia"}:
            label_token = clean_var_token(source.label)
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
        attrs=build_state_attrs(
            species_name=",".join(_composition_species_names(str(args.composition))),
            temperature_k=spectrum.temperature_k,
            pressure_pa=spectrum.pressure_pa,
            extra={"composition_input": str(args.composition), **_band_attr_values(args.wn_range)},
        ),
    )
    add_wavenumber_attrs(dataset)
    return dataset


def _species_transmission_dataset(
    *,
    spectrum,
    transmittance,
    species_name: str,
    secondary_component: dict[str, object] | None = None,
    wn_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    species_token = clean_var_token(species_name)
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
        if str(secondary_component.get("kind", "")) == "continuum":
            label_token = _continuum_label_token(label)
            trans_name = f"transmittance_continuum_{label_token}"
            att_name = f"attenuation_continuum_{label_token}"
            trans_long = f"{label} transmittance"
            att_long = f"{label} attenuation coefficient"
        else:
            label_token = clean_var_token(label)
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
        attrs=build_state_attrs(
            species_name=str(species_name),
            temperature_k=transmittance.temperature_k,
            pressure_pa=transmittance.pressure_pa,
            extra={"path_length_m": float(transmittance.path_length_m), **({} if wn_range is None else _band_attr_values(wn_range))},
        ),
    )
    add_wavenumber_attrs(dataset)
    return dataset


def _pair_transmission_dataset(args: argparse.Namespace) -> xr.Dataset:
    spectrum = _compute_pair_xsection(args)
    transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=float(args.path_length_km) * 1000.0)
    pair, _ = _resolve_pair_filename(args)
    pair_token = clean_var_token(pair)
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
        attrs=build_state_attrs(
            species_name=pair,
            temperature_k=transmittance.temperature_k,
            pressure_pa=transmittance.pressure_pa,
            extra={"path_length_m": float(transmittance.path_length_m), **_band_attr_values(args.wn_range)},
        ),
    )
    add_wavenumber_attrs(dataset)
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
        species_token = clean_var_token(term.species_name)
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
        attenuation = np.asarray(source.sigma_cm2_molecule * number_density_cm3 * 100.0, dtype=np.float64)
        if source.kind == "continuum":
            label_token = _continuum_label_token(source.label)
            att_name = f"attenuation_continuum_{label_token}"
            trans_name = f"transmittance_continuum_{label_token}"
            prefix = f"{source.label} weighted"
        else:
            label_token = clean_var_token(source.label)
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
        attrs=build_state_attrs(
            species_name=",".join(_composition_species_names(str(args.composition))),
            temperature_k=transmittance.temperature_k,
            pressure_pa=transmittance.pressure_pa,
            extra={"composition_input": str(args.composition), "path_length_m": path_length_m, **_band_attr_values(args.wn_range)},
        ),
    )
    add_wavenumber_attrs(dataset)
    return dataset


def _write_xsection_dataset(
    spectrum,
    output_path: Path,
    *,
    species_name: str | None = None,
    secondary_component: dict[str, object] | None = None,
    wn_range: tuple[float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = _xsection_dataset(spectrum, species_name=species_name, secondary_component=secondary_component, wn_range=wn_range)
    try:
        write_dataset_via_tmp(dataset, output_path, engine=DEFAULT_NETCDF_ENGINE)
    finally:
        dataset.close()


def _resolve_pair_filename(args: argparse.Namespace) -> tuple[str, str]:
    pair = str(args.pair or "H2-H2")
    if args.filename:
        return pair, str(args.filename)
    if getattr(args, "cia_database", "hitran") == "orton_xiz":
        legacy_model = "xiz" if args.cia_model == "auto" else args.cia_model
        return pair, resolve_orton_xiz_cia_filename(pair=pair, model=legacy_model, state=args.cia_state)
    metadata = resolve_hitran_cia_filename(pair=pair, model=args.cia_model, state=args.cia_state)
    return metadata.pair, metadata.filename


def _load_pair_cia_dataset(args: argparse.Namespace):
    pair, filename = _resolve_pair_filename(args)
    if getattr(args, "cia_database", "hitran") == "orton_xiz":
        return load_orton_xiz_cia_dataset(
            cache_dir=args.cia_dir or default_orton_xiz_cia_dir(),
            pair=pair,
            model="xiz" if args.cia_model == "auto" else args.cia_model,
            state=args.cia_state,
            refresh=bool(args.refresh_cia),
        )
    return load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=filename,
        pair=pair,
        index_url=str(args.cia_index_url),
        refresh=bool(args.refresh_cia),
    )


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
    temperature_k, pressure_bar = _single_base_state(args)
    pressure_pa = pressure_bar * 1.0e5
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
    pair, _ = _resolve_pair_filename(args)
    band = build_band(args)
    cia_dataset = _load_pair_cia_dataset(args)
    spectrum = compute_absorption_spectrum_from_sources(
        species_name=pair,
        wavenumber_grid_cm1=band.grid(),
        temperature_k=_single_base_state(args)[0],
        pressure_pa=_single_base_state(args)[1] * 1.0e5,
        line_provider=_ZeroLineProvider(),
        cia_dataset=cia_dataset,
    )
    return spectrum


def _pair_xsection_dataset(args: argparse.Namespace) -> xr.Dataset:
    pair, filename = _resolve_pair_filename(args)
    band = build_band(args)
    cia_dataset = _load_pair_cia_dataset(args)
    grid = band.grid()
    temperature_k, _ = _single_base_state(args)
    binary = np.asarray(
        cia_dataset.interpolate_to_grid(
            temperature_k=temperature_k,
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
            "temperature_k": temperature_k,
            "source_filename": filename,
            **_band_attr_values(args.wn_range),
        },
    )


def _compute_composition_products(args: argparse.Namespace):
    temperature_k, pressure_bar = _single_base_state(args)
    mixture_args = argparse.Namespace(
        composition=args.composition,
        hitran_dir=args.hitran_dir,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        resolution=args.resolution,
        cia_index_url=args.cia_index_url,
        refresh_hitran=args.refresh_hitran,
        refresh_cia=args.refresh_cia,
        broadening_composition=args.broadening_composition,
        path_length_km=getattr(args, "path_length_km", 1.0),
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
            wn_range=args.wn_range,
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
        transmittance = compute_transmittance_spectrum(spectrum=spectrum, path_length_m=float(args.path_length_km) * 1000.0)
        dataset = _species_transmission_dataset(
            spectrum=spectrum,
            transmittance=transmittance,
            species_name=str(args.species or "CO2"),
            secondary_component=secondary_component,
            wn_range=args.wn_range,
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
    ctx = process_pool_context()
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        return list(executor.map(worker, tasks))


def _stack_state_grid_datasets(
    datasets: list[xr.Dataset],
    *,
    base_temperatures: list[float],
    pressure_bars: list[float],
    del_temperatures: list[float],
) -> xr.Dataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    per_pressure: list[xr.Dataset] = []
    index = 0
    for _ in pressure_bars:
        del_slice = datasets[index:index + len(del_temperatures)]
        index += len(del_temperatures)
        per_pressure.append(
            xr.concat(
                del_slice,
                dim=xr.IndexVariable("del_temperature", np.asarray(del_temperatures, dtype=np.float64)),
            )
        )
    stacked = xr.concat(
        per_pressure,
        dim=xr.IndexVariable("pressure", np.asarray(pressure_bars, dtype=np.float64) * 1.0e5),
    ).transpose("del_temperature", "pressure", "wavenumber")
    stacked["del_temperature"].attrs = dict(DEL_TEMPERATURE_COORD_ATTRS)
    stacked["pressure"].attrs = dict(PRESSURE_COORD_ATTRS)
    stacked["temperature"] = (
        "pressure",
        np.asarray(base_temperatures, dtype=np.float64),
        dict(TEMPERATURE_VAR_ATTRS),
    )
    attrs = dict(datasets[0].attrs)
    attrs.pop("temperature_k", None)
    attrs.pop("pressure_pa", None)
    attrs.pop("pressure_bar", None)
    attrs.pop("number_density_cm3", None)
    stacked.attrs = attrs
    return stacked


def _compute_range_temperature_datasets(
    *,
    args: argparse.Namespace,
    wn_ranges: list[tuple[float, float]],
    target_kind: str,
    worker,
) -> list[tuple[tuple[float, float], xr.Dataset, list[str]]]:
    state_pairs = _state_pairs(args)
    base_temperatures = [temperature_k for temperature_k, _ in state_pairs]
    pressure_bars = [pressure_bar for _, pressure_bar in state_pairs]
    del_temperatures = _selected_del_temperatures(args)
    tasks: list[tuple[str, argparse.Namespace]] = []
    for wn_range in wn_ranges:
        range_args = _args_for_wn_range(args, wn_range)
        for base_temperature_k, pressure_bar in state_pairs:
            for del_temperature_k in del_temperatures:
                tasks.append(
                    (
                        target_kind,
                        _args_for_state(
                            range_args,
                            base_temperature_k=base_temperature_k,
                            pressure_bar=pressure_bar,
                            del_temperature_k=del_temperature_k,
                        ),
                    )
                )

    results = _parallel_band_results(tasks, worker=worker)
    grouped: list[tuple[tuple[float, float], xr.Dataset, list[str]]] = []
    index = 0
    for wn_range in wn_ranges:
        span = len(base_temperatures) * len(del_temperatures)
        range_results = results[index:index + span]
        index += span
        datasets = [dataset for dataset, _ in range_results]
        broadening_summaries = [summary for _, summary in range_results if summary]
        try:
            stacked = _stack_state_grid_datasets(
                datasets,
                base_temperatures=base_temperatures,
                pressure_bars=pressure_bars,
                del_temperatures=del_temperatures,
            )
        finally:
            for dataset in datasets:
                dataset.close()
        grouped.append((wn_range, stacked, broadening_summaries))
    return grouped


def build_parser() -> argparse.ArgumentParser:
    """Create the dump CLI parser."""
    parser = DumpArgumentParser(
        description="Write spectroscopy NetCDF products from HITRAN line and CIA data.",
        formatter_class=HelpFormatter,
        epilog=dedent(
            """\
            Examples:
              pyharp-dump xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump xsection --pair H2-He --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump transmission --composition H2:0.9,He:0.1,CH4:0.004 --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

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
              pyharp-dump transmission --species CO2 --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
              pyharp-dump transmission --pair H2-H2 --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
              pyharp-dump transmission --composition H2:0.9,He:0.1,CH4:0.004 --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
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
    _validate_state_grid(args, parser)
    if args.command == "xsection":
        wn_ranges = _selected_wn_ranges(args)
        target_kind, _ = _selected_target(args)
        for wn_range, dataset, broadening_summaries in _compute_range_temperature_datasets(
            args=args,
            wn_ranges=wn_ranges,
            target_kind=target_kind,
            worker=_compute_xsection_band,
        ):
            output_path = _output_path_for_wn_range(args, wn_range=wn_range, suffix=".nc")
            try:
                write_dataset_via_tmp(dataset, output_path, engine=DEFAULT_NETCDF_ENGINE)
            finally:
                dataset.close()
            print(f"Wrote NetCDF: {output_path}")
            for broadening_summary in broadening_summaries:
                print(f"Broadening: {broadening_summary}")
        return
    if args.command == "transmission":
        wn_ranges = _selected_wn_ranges(args)
        target_kind, _ = _selected_target(args)
        for wn_range, dataset, broadening_summaries in _compute_range_temperature_datasets(
            args=args,
            wn_ranges=wn_ranges,
            target_kind=target_kind,
            worker=_compute_transmission_band,
        ):
            output_path = _output_path_for_wn_range(args, wn_range=wn_range, suffix=".nc")
            try:
                write_dataset_via_tmp(dataset, output_path, engine=DEFAULT_NETCDF_ENGINE)
            finally:
                dataset.close()
            print(f"Wrote NetCDF: {output_path}")
            for broadening_summary in broadening_summaries:
                print(f"Broadening: {broadening_summary}")
        return
