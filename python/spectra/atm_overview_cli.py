"""Shared CLI helpers for atmospheric mixture overview plotting."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectra_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .config import (
    SpectroscopyConfig,
    SpectralBandConfig,
    parse_broadening_composition,
    resolve_hitran_species,
    supported_hitran_cia_pairs,
    supported_hitran_species_names,
)
from .hitran_cia import load_cia_dataset
from .hitran_lines import build_line_provider, download_hitran_lines, load_hitran_line_list
from .molecule_plot_cli import _add_legend_if_needed, _apply_positive_log_scale, _mask_nonpositive, _plot_transmittance_panel, _style_shared_axis
from .mt_ckd_h2o import compute_mt_ckd_h2o_continuum_cross_section
from .shared_cli import process_pool_context
from .spectrum import AbsorptionSpectrum, number_density_cm3_from_pressure_temperature
from .transmittance import compute_transmittance_spectrum


@dataclass(frozen=True)
class MixtureSpeciesTerm:
    species_name: str
    mole_fraction: float
    table_name: str
    sigma_line_cm2_molecule: np.ndarray
    line_list_wavenumber_cm1: np.ndarray
    line_list_strength: np.ndarray


@dataclass(frozen=True)
class MixtureSecondarySource:
    kind: str
    label: str
    weight: float
    source_name: str
    sigma_cm2_molecule: np.ndarray


@dataclass(frozen=True)
class MixtureOverviewProducts:
    band: SpectralBandConfig
    composition: dict[str, float]
    species_terms: tuple[MixtureSpeciesTerm, ...]
    secondary_sources: tuple[MixtureSecondarySource, ...]
    spectrum: AbsorptionSpectrum
    transmittance: object
    manifest_sources: tuple[dict[str, object], ...]


def _canonical_mixture_species_names() -> dict[str, str]:
    mapping: dict[str, str] = {name.upper(): name for name in supported_hitran_species_names()}
    for metadata in supported_hitran_cia_pairs():
        first, second = metadata.pair.split("-")
        mapping[first.upper()] = first
        mapping[second.upper()] = second
    return mapping


def _parse_composition(value: str) -> dict[str, float]:
    canonical_names = _canonical_mixture_species_names()
    entries: list[tuple[str, float]] = []
    for chunk in str(value).split(","):
        piece = chunk.strip()
        if not piece:
            continue
        name, sep, fraction_text = piece.partition(":")
        if not sep:
            raise ValueError("composition entries must have the form SPECIES:FRACTION")
        key = name.strip().upper()
        try:
            species_name = canonical_names[key]
        except KeyError as exc:
            supported = ", ".join(sorted(canonical_names.values()))
            raise ValueError(f"Unsupported composition species {name!r}. Supported names: {supported}.") from exc
        try:
            fraction = float(fraction_text)
        except ValueError as exc:
            raise ValueError(f"Invalid mole fraction {fraction_text!r} for species {name!r}.") from exc
        if fraction < 0.0:
            raise ValueError("composition fractions must be non-negative")
        entries.append((species_name, fraction))
    if not entries:
        raise ValueError("composition must contain at least one SPECIES:FRACTION entry")
    totals: dict[str, float] = {}
    for species_name, fraction in entries:
        totals[species_name] = totals.get(species_name, 0.0) + fraction
    total_fraction = sum(totals.values())
    if total_fraction <= 0.0:
        raise ValueError("composition fractions must sum to a positive value")
    return {species_name: value / total_fraction for species_name, value in totals.items()}


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


def _build_band(*, wn_min: float, wn_max: float, resolution: float) -> SpectralBandConfig:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if wn_max < wn_min:
        raise ValueError("wn-range max must be >= min")
    return SpectralBandConfig("single_state", float(wn_min), float(wn_max), float(resolution))


def _build_species_config(
    *,
    species_name: str,
    band: SpectralBandConfig,
    hitran_dir: Path,
    refresh_hitran: bool,
    broadening_composition: dict[str, float] | None = None,
) -> SpectroscopyConfig:
    return SpectroscopyConfig(
        output_path=Path("output") / "unused.nc",
        hitran_cache_dir=hitran_dir,
        species_name=species_name,
        broadening_composition=broadening_composition,
        refresh_hitran=refresh_hitran,
    )


def _find_binary_pairs(composition: dict[str, float]) -> tuple[tuple[str, str, str, str], ...]:
    species_set = set(composition)
    pairs: list[tuple[str, str, str, str]] = []
    for metadata in supported_hitran_cia_pairs():
        first, second = metadata.pair.split("-")
        if first == second:
            continue
        if first in species_set and second in species_set:
            pairs.append((metadata.pair, metadata.filename, first, second))
    return tuple(pairs)


def _line_supported_species(composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
    supported = set(supported_hitran_species_names())
    return tuple((name, composition[name]) for name in composition if name in supported)


def compute_mixture_overview_products(args: argparse.Namespace, *, wn_range: tuple[float, float]) -> MixtureOverviewProducts:
    if args.path_length_km <= 0.0:
        raise ValueError("path-length-km must be positive")
    composition = _parse_composition(args.composition)
    broadening_composition = parse_broadening_composition(
        getattr(args, "broadening_composition", None) or composition
    )
    band = _build_band(wn_min=wn_range[0], wn_max=wn_range[1], resolution=float(args.resolution))
    grid = band.grid()
    temperature_k = float(args.temperature_k)
    pressure_pa = float(args.pressure_bar) * 1.0e5
    number_density_cm3 = number_density_cm3_from_pressure_temperature(pressure_pa=pressure_pa, temperature_k=temperature_k)

    species_terms: list[MixtureSpeciesTerm] = []
    line_total = np.zeros(grid.shape, dtype=np.float64)
    secondary_total = np.zeros(grid.shape, dtype=np.float64)
    manifest_sources: list[dict[str, object]] = []

    for species_name, mole_fraction in _line_supported_species(composition):
        config = _build_species_config(
            species_name=species_name,
            band=band,
            hitran_dir=args.hitran_dir,
            broadening_composition=broadening_composition,
            refresh_hitran=bool(args.refresh_hitran),
        )
        line_db = download_hitran_lines(config, band)
        line_provider = build_line_provider(config, line_db)
        print(f"{species_name} broadening: {line_provider.broadening_summary()}")
        sigma_line = np.asarray(
            line_provider.cross_section_cm2_molecule(
                wavenumber_grid_cm1=grid,
                temperature_k=temperature_k,
                pressure_pa=pressure_pa,
            ),
            dtype=np.float64,
        )
        line_list = load_hitran_line_list(config, band)
        species_terms.append(
            MixtureSpeciesTerm(
                species_name=species_name,
                mole_fraction=mole_fraction,
                table_name=line_db.table_name,
                sigma_line_cm2_molecule=sigma_line,
                line_list_wavenumber_cm1=np.asarray(line_list.wavenumber_cm1, dtype=np.float64),
                line_list_strength=np.asarray(line_list.line_intensity, dtype=np.float64),
            )
        )
        line_total += mole_fraction * sigma_line
        manifest_sources.append(
            {
                "kind": "line",
                "species": species_name,
                "mole_fraction": mole_fraction,
                "table_name": line_db.table_name,
            }
        )

    secondary_sources: list[MixtureSecondarySource] = []

    if "H2O" in composition:
        h2o_fraction = composition["H2O"]
        sigma_continuum = np.asarray(
            compute_mt_ckd_h2o_continuum_cross_section(
                wavenumber_grid_cm1=grid,
                temperature_k=temperature_k,
                pressure_pa=pressure_pa,
                h2o_vmr=h2o_fraction,
            ),
            dtype=np.float64,
        )
        secondary_total += sigma_continuum
        secondary_sources.append(
            MixtureSecondarySource(
                kind="continuum",
                label="H2O continuum (MT_CKD)",
                weight=h2o_fraction,
                source_name="MT_CKD_H2O",
                sigma_cm2_molecule=sigma_continuum,
            )
        )
        manifest_sources.append(
            {
                "kind": "continuum",
                "species": "H2O",
                "mole_fraction": h2o_fraction,
                "source_name": "MT_CKD_H2O",
            }
        )

    for species_name, mole_fraction in composition.items():
        if species_name not in set(supported_hitran_species_names()):
            continue
        config = _build_species_config(
            species_name=species_name,
            band=band,
            hitran_dir=args.hitran_dir,
            refresh_hitran=bool(args.refresh_hitran),
        )
        cia_filename = config.hitran_species.cia_filename
        if cia_filename is None or species_name == "H2O":
            continue
        cia_dataset = load_cia_dataset(
            cache_dir=args.hitran_dir,
            filename=cia_filename,
            pair=config.cia_pair,
            index_url=str(args.cia_index_url),
            refresh=bool(args.refresh_cia),
        )
        binary = cia_dataset.interpolate_to_grid(temperature_k, grid)
        sigma_secondary = np.asarray(binary, dtype=np.float64) * (mole_fraction ** 2) * number_density_cm3
        secondary_total += sigma_secondary
        secondary_sources.append(
            MixtureSecondarySource(
                kind="self_cia",
                label=config.cia_pair,
                weight=mole_fraction ** 2,
                source_name=str(cia_dataset.source_path),
                sigma_cm2_molecule=sigma_secondary,
            )
        )
        manifest_sources.append(
            {
                "kind": "self_cia",
                "pair": config.cia_pair,
                "weight": mole_fraction ** 2,
                "source_name": str(cia_dataset.source_path),
            }
        )

    for pair_name, filename, first, second in _find_binary_pairs(composition):
        cia_dataset = load_cia_dataset(
            cache_dir=args.hitran_dir,
            filename=filename,
            pair=pair_name,
            index_url=str(args.cia_index_url),
            refresh=bool(args.refresh_cia),
        )
        binary = cia_dataset.interpolate_to_grid(temperature_k, grid)
        pair_weight = composition[first] * composition[second]
        sigma_secondary = np.asarray(binary, dtype=np.float64) * pair_weight * number_density_cm3
        secondary_total += sigma_secondary
        secondary_sources.append(
            MixtureSecondarySource(
                kind="binary_cia",
                label=pair_name,
                weight=pair_weight,
                source_name=str(cia_dataset.source_path),
                sigma_cm2_molecule=sigma_secondary,
            )
        )
        manifest_sources.append(
            {
                "kind": "binary_cia",
                "pair": pair_name,
                "weight": pair_weight,
                "source_name": str(cia_dataset.source_path),
            }
        )

    kappa_line_cm1 = line_total * number_density_cm3
    kappa_secondary_cm1 = secondary_total * number_density_cm3
    spectrum = AbsorptionSpectrum(
        species_name=str(args.composition),
        wavenumber_cm1=np.asarray(grid, dtype=np.float64),
        sigma_line_cm2_molecule=np.asarray(line_total, dtype=np.float64),
        sigma_cia_cm2_molecule=np.asarray(secondary_total, dtype=np.float64),
        sigma_total_cm2_molecule=np.asarray(line_total + secondary_total, dtype=np.float64),
        kappa_line_cm1=np.asarray(kappa_line_cm1, dtype=np.float64),
        kappa_cia_cm1=np.asarray(kappa_secondary_cm1, dtype=np.float64),
        kappa_total_cm1=np.asarray(kappa_line_cm1 + kappa_secondary_cm1, dtype=np.float64),
        attenuation_line_m1=np.asarray(kappa_line_cm1 * 100.0, dtype=np.float64),
        attenuation_cia_m1=np.asarray(kappa_secondary_cm1 * 100.0, dtype=np.float64),
        attenuation_total_m1=np.asarray((kappa_line_cm1 + kappa_secondary_cm1) * 100.0, dtype=np.float64),
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        number_density_cm3=number_density_cm3,
    )
    transmittance = compute_transmittance_spectrum(
        spectrum=spectrum,
        path_length_m=float(args.path_length_km) * 1000.0,
    )
    return MixtureOverviewProducts(
        band=band,
        composition=composition,
        species_terms=tuple(species_terms),
        secondary_sources=tuple(secondary_sources),
        spectrum=spectrum,
        transmittance=transmittance,
        manifest_sources=tuple(manifest_sources),
    )


def _plot_mixture_line_panel(ax, species_terms: tuple[MixtureSpeciesTerm, ...], *, xlim: tuple[float, float], min_line_strength: float = 1.0e-27) -> None:
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(species_terms), 1)))
    any_lines = False
    for color, term in zip(colors, species_terms):
        mask = (
            (term.line_list_wavenumber_cm1 >= xlim[0])
            & (term.line_list_wavenumber_cm1 <= xlim[1])
            & (term.line_list_strength >= float(min_line_strength))
        )
        wavenumber = term.line_list_wavenumber_cm1[mask]
        strength = term.line_list_strength[mask]
        if wavenumber.size == 0:
            continue
        any_lines = True
        ax.vlines(wavenumber, float(min_line_strength), strength, color=color, alpha=0.1, linewidth=0.35)
        ax.scatter(wavenumber, strength, s=4.0, color=color, alpha=0.65, linewidths=0.0, label=term.species_name)
    if not any_lines:
        ax.text(0.5, 0.5, "No line strengths above the minimum threshold in range.", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Line strength", fontsize=9)
        _style_shared_axis(ax, xlim=xlim)
        return
    ax.set_yscale("log")
    ax.set_ylabel("Line strength\n[cm$^{-1}$/ (molecule cm$^{-2}$)]", fontsize=9)
    ax.set_title("Molecular line positions and strengths", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)
    ax.legend(loc="upper right", fontsize=8, ncol=2)


def _plot_source_cross_section_panel(ax, products: MixtureOverviewProducts, *, xlim: tuple[float, float]) -> None:
    color_values = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(products.species_terms) + len(products.secondary_sources), 1)))
    color_index = 0
    series: list[np.ndarray] = [products.spectrum.sigma_total_cm2_molecule]
    for term in products.species_terms:
        weighted_sigma = term.mole_fraction * term.sigma_line_cm2_molecule
        series.append(weighted_sigma)
        ax.plot(products.spectrum.wavenumber_cm1, _mask_nonpositive(weighted_sigma), color=color_values[color_index], linewidth=1.0, label=f"{term.species_name} line")
        color_index += 1
    for source in products.secondary_sources:
        series.append(source.sigma_cm2_molecule)
        ax.plot(products.spectrum.wavenumber_cm1, _mask_nonpositive(source.sigma_cm2_molecule), color=color_values[color_index], linewidth=1.0, label=source.label)
        color_index += 1
    ax.plot(products.spectrum.wavenumber_cm1, _mask_nonpositive(products.spectrum.sigma_total_cm2_molecule), color="black", linewidth=1.5, label="Total")
    if not _apply_positive_log_scale(ax, series):
        ax.text(0.5, 0.5, "No positive cross-section values.", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Cross section\n[cm$^2$ / molecule]", fontsize=9)
    ax.set_title("Absorption cross section", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)
    _add_legend_if_needed(ax)


def _plot_source_attenuation_panel(ax, products: MixtureOverviewProducts, *, xlim: tuple[float, float]) -> None:
    color_values = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(products.species_terms) + len(products.secondary_sources), 1)))
    color_index = 0
    series: list[np.ndarray] = [products.spectrum.attenuation_total_m1]
    factor = products.spectrum.number_density_cm3 * 100.0
    for term in products.species_terms:
        attenuation = term.mole_fraction * term.sigma_line_cm2_molecule * factor
        series.append(attenuation)
        ax.plot(products.spectrum.wavenumber_cm1, _mask_nonpositive(attenuation), color=color_values[color_index], linewidth=1.0, label=f"{term.species_name} line")
        color_index += 1
    for source in products.secondary_sources:
        attenuation = source.sigma_cm2_molecule * factor
        series.append(attenuation)
        ax.plot(products.spectrum.wavenumber_cm1, _mask_nonpositive(attenuation), color=color_values[color_index], linewidth=1.0, label=source.label)
        color_index += 1
    ax.plot(products.spectrum.wavenumber_cm1, _mask_nonpositive(products.spectrum.attenuation_total_m1), color="black", linewidth=1.5, label="Total")
    if not _apply_positive_log_scale(ax, series):
        ax.text(0.5, 0.5, "No positive attenuation values.", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Attenuation\n[1 / m]", fontsize=9)
    ax.set_title("Attenuation coefficient", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)
    _add_legend_if_needed(ax)


def _render_mixture_overview(fig, axes_flat, *, products: MixtureOverviewProducts) -> None:
    band = products.band
    xlim = (band.wavenumber_min_cm1, band.wavenumber_max_cm1)
    fig.subplots_adjust(
        left=1.0 / 8.5,
        right=1.0 - 1.0 / 8.5,
        bottom=1.0 / 11.0,
        top=1.0 - 1.2 / 11.0,
        hspace=0.5,
    )
    comp_text = ", ".join(f"{name}:{value:.3f}" for name, value in products.composition.items())
    fig.text(
        1.0 / 8.5,
        1.0 - 0.32 / 11.0,
        (
            f"Atmosphere overview | {comp_text}\n"
            f"{band.wavenumber_min_cm1:.3f}-{band.wavenumber_max_cm1:.3f} cm$^{{-1}}$ | "
            f"T={products.spectrum.temperature_k:.1f} K | p={products.spectrum.pressure_pa / 1.0e5:.3f} bar | "
            f"L={products.transmittance.path_length_m / 1000.0:.3f} km"
        ),
        ha="left",
        va="center",
        fontsize=11,
    )
    _plot_mixture_line_panel(axes_flat[0], products.species_terms, xlim=xlim)
    _plot_source_cross_section_panel(axes_flat[1], products, xlim=xlim)
    _plot_source_attenuation_panel(axes_flat[2], products, xlim=xlim)
    _plot_transmittance_panel(axes_flat[3], products.transmittance, xlim=xlim, show_components=False, component_label="Secondary")
    for ax in axes_flat:
        ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontsize=10)


def run_atm_attenuation(args: argparse.Namespace, *, wn_range: tuple[float, float]) -> None:
    products = compute_mixture_overview_products(args, wn_range=wn_range)
    figure_path = args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xlim = (products.band.wavenumber_min_cm1, products.band.wavenumber_max_cm1)
    _plot_source_attenuation_panel(ax, products, xlim=xlim)
    ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Atmosphere attenuation figure: {figure_path}")
    print(f"Composition: {args.composition}")
    print(f"Grid points: {products.spectrum.wavenumber_cm1.size}")


def run_atm_transmission(args: argparse.Namespace, *, wn_range: tuple[float, float]) -> None:
    products = compute_mixture_overview_products(args, wn_range=wn_range)
    figure_path = args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xlim = (products.band.wavenumber_min_cm1, products.band.wavenumber_max_cm1)
    _plot_transmittance_panel(ax, products.transmittance, xlim=xlim, show_components=False, component_label="Secondary")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Atmosphere transmission figure: {figure_path}")
    print(f"Composition: {args.composition}")
    print(f"Grid points: {products.spectrum.wavenumber_cm1.size}")


def _page_manifest(products: MixtureOverviewProducts) -> dict[str, object]:
    return {
        "wavenumber_min_cm1": products.band.wavenumber_min_cm1,
        "wavenumber_max_cm1": products.band.wavenumber_max_cm1,
        "resolution_cm1": products.band.resolution_cm1,
        "grid_points": int(products.spectrum.wavenumber_cm1.size),
        "temperature_k": float(products.spectrum.temperature_k),
        "pressure_bar": float(products.spectrum.pressure_pa) / 1.0e5,
        "line_species": [term.species_name for term in products.species_terms],
        "opacity_sources": list(products.manifest_sources),
    }


def _selected_temperatures(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "temperature_k", [300.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _selected_pressure_bars(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "pressure_bar", [1.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _state_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    temperatures = _selected_temperatures(args)
    pressures = _selected_pressure_bars(args)
    if len(temperatures) != len(pressures):
        raise ValueError("temperature_k and pressure_bar must have the same number of values")
    return list(zip(temperatures, pressures, strict=True))


def build_atm_overview_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a 4-row atmospheric opacity overview PDF for a gas mixture.")
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"))
    parser.add_argument("--composition", required=True)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--path-length-km", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--wn-range", dest="wn_ranges", action="append", type=_parse_wn_range, required=True)
    parser.add_argument("--broadening-composition", default=None)
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/")
    parser.add_argument("--refresh-hitran", action="store_true")
    parser.add_argument("--refresh-cia", action="store_true")
    parser.add_argument("--figure", type=Path, default=Path("output/atm_overview.pdf"))
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main_atm_overview() -> None:
    args = build_atm_overview_parser().parse_args()
    run_atm_overview(args)


def run_atm_overview(args: argparse.Namespace) -> None:
    figure_path = args.figure
    manifest_path = args.manifest or figure_path.with_suffix(".manifest.json")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    wn_ranges = list(args.wn_ranges)
    state_pairs = _state_pairs(args)
    tasks: list[tuple[argparse.Namespace, tuple[float, float]]] = []
    page_metadata: list[tuple[float, float, tuple[float, float]]] = []
    for temperature_k, pressure_bar in state_pairs:
        state_args = argparse.Namespace(**vars(args))
        state_args.temperature_k = float(temperature_k)
        state_args.pressure_bar = float(pressure_bar)
        for wn_range in wn_ranges:
            tasks.append((state_args, wn_range))
            page_metadata.append((float(temperature_k), float(pressure_bar), wn_range))
    pages: list[dict[str, object]] = []
    with PdfPages(figure_path) as pdf:
        for (temperature_k, pressure_bar, wn_range), products in zip(
            page_metadata,
            _parallel_mixture_overview_products(tasks),
            strict=True,
        ):
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8.5, 11.0), squeeze=False)
            _render_mixture_overview(fig, axes[:, 0], products=products)
            pdf.savefig(fig)
            plt.close(fig)
            pages.append(_page_manifest(products))
            print(f"Added page: T={temperature_k:.1f} K | p={pressure_bar:.3f} bar | {wn_range[0]:.3f}-{wn_range[1]:.3f} cm^-1")

    manifest = {
        "composition_input": str(args.composition),
        "composition_normalized": _parse_composition(args.composition),
        "temperature_k": _selected_temperatures(args),
        "pressure_bar": _selected_pressure_bars(args),
        "path_length_km": float(args.path_length_km),
        "figure_path": str(figure_path),
        "manifest_path": str(manifest_path),
        "pages": pages,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"Atmosphere overview figure: {figure_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Pages: {len(pages)}")


def _compute_mixture_overview_product_task(task: tuple[argparse.Namespace, tuple[float, float]]) -> MixtureOverviewProducts:
    args, wn_range = task
    return compute_mixture_overview_products(args, wn_range=wn_range)


def _parallel_mixture_overview_products(
    tasks: list[tuple[argparse.Namespace, tuple[float, float]]],
) -> Iterator[MixtureOverviewProducts]:
    if len(tasks) <= 1:
        for task in tasks:
            yield _compute_mixture_overview_product_task(task)
        return
    max_workers = min(len(tasks), os.cpu_count() or 1)
    ctx = process_pool_context()
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        yield from executor.map(_compute_mixture_overview_product_task, tasks)
