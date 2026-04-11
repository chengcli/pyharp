"""Shared CLI helpers for molecular absorption plotting scripts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import tempfile
import multiprocessing as mp

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectra_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .blackbody import compute_normalized_blackbody_curve
from .config import SpectroscopyConfig, SpectralBandConfig, parse_broadening_composition, resolve_hitran_cia_pair
from .hitran_cia import load_cia_dataset
from .hitran_lines import LineDatabase, build_line_provider, download_hitran_lines, load_hitran_line_list, plot_hitran_line_positions
from .spectrum import _resolve_continuum_sources, compute_absorption_spectrum_from_sources, plot_absorption_spectrum, plot_attenuation_spectrum
from .transmittance import compute_transmittance_spectrum, plot_transmittance_spectrum


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


def _wn_bounds(args: argparse.Namespace) -> tuple[float, float]:
    return tuple(args.wn_range)


def _add_common_arguments(parser: argparse.ArgumentParser, *, include_state: bool = True) -> None:
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"))
    parser.add_argument("--species", default="H2O")
    if include_state:
        parser.add_argument("--temperature-k", type=float, default=300.0)
        parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--wn-range", type=_parse_wn_range, default=(20.0, 2500.0))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--refresh-hitran", action="store_true")
    parser.add_argument("--broadening-composition", default=None)


def _add_external_cia_arguments(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--cia-filename", required=required)
    parser.add_argument("--cia-pair", default=None)
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/")
    parser.add_argument("--refresh-cia", action="store_true")


def _build_band_and_config(args: argparse.Namespace) -> tuple[SpectralBandConfig, SpectroscopyConfig]:
    wn_min, wn_max = _wn_bounds(args)
    if args.resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if wn_max < wn_min:
        raise ValueError("wn-range max must be >= min")
    band = SpectralBandConfig("single_state", wn_min, wn_max, args.resolution)
    config = SpectroscopyConfig(
        output_path=Path("output") / "unused.nc",
        hitran_cache_dir=args.hitran_dir,
        species_name=args.species,
        broadening_composition=parse_broadening_composition(getattr(args, "broadening_composition", None)),
        refresh_hitran=args.refresh_hitran,
    )
    return band, config


def _print_positive_summary(values: np.ndarray, label: str, unit: str) -> None:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        print(f"{label}: all values are zero")
        return
    print(f"{label} min/max: {positive.min():.3e} .. {positive.max():.3e} {unit}")


def _print_bounded_summary(values: np.ndarray, label: str) -> None:
    bounded = np.asarray(values, dtype=np.float64)
    print(f"{label} min/max: {bounded.min():.3e} .. {bounded.max():.3e}")


def _resolve_cia_pair(args: argparse.Namespace, config: SpectroscopyConfig) -> str:
    return str(args.cia_pair or f"{config.hitran_species.name}-{config.hitran_species.name}")


def _resolve_self_component_filename(args: argparse.Namespace, config: SpectroscopyConfig) -> str | None:
    explicit = getattr(args, "cia_filename", None)
    if explicit:
        return str(explicit)
    explicit_pair = getattr(args, "cia_pair", None)
    if explicit_pair:
        return resolve_hitran_cia_pair(explicit_pair).filename
    return config.hitran_species.cia_filename


def _load_requested_cia_dataset(args: argparse.Namespace, config: SpectroscopyConfig):
    cia_filename = _resolve_self_component_filename(args, config)
    if cia_filename is None:
        return None
    return load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=cia_filename,
        pair=_resolve_cia_pair(args, config),
        index_url=str(args.cia_index_url),
        refresh=bool(args.refresh_cia),
    )


_MISSING = object()


def _compute_requested_absorption_spectrum(
    args: argparse.Namespace,
    *,
    line_db: LineDatabase | None = None,
    cia_dataset=_MISSING,
):
    band, config = _build_band_and_config(args)
    temperature_k = float(args.temperature_k)
    pressure_pa = float(args.pressure_bar) * 1.0e5
    if cia_dataset is _MISSING:
        cia_dataset = _load_requested_cia_dataset(args, config)
    line_db = line_db or download_hitran_lines(config, band)
    line_provider = build_line_provider(config, line_db)
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
    return band, config, spectrum, line_provider


def _print_spectrum_summary(spectrum, *, include_components: bool = False) -> None:
    print(f"Species: {spectrum.species_name}")
    print(f"Grid points: {spectrum.wavenumber_cm1.size}")
    if include_components:
        _print_positive_summary(spectrum.attenuation_line_m1, "Line attenuation", "1/m")
        _print_positive_summary(spectrum.attenuation_cia_m1, "CIA / Continuum attenuation", "1/m")
        _print_positive_summary(spectrum.attenuation_total_m1, "Total attenuation", "1/m")


def _print_transmittance_summary(transmittance, *, include_components: bool = False) -> None:
    print(f"Species: {transmittance.species_name}")
    print(f"Grid points: {transmittance.wavenumber_cm1.size}")
    if include_components:
        _print_bounded_summary(transmittance.transmittance_line, "Line transmission")
        _print_bounded_summary(transmittance.transmittance_cia, "CIA / Continuum transmission")
    _print_bounded_summary(transmittance.transmittance_total, "Total transmission")


def build_xsection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute and plot molecular absorption cross section.")
    _add_common_arguments(parser)
    _add_external_cia_arguments(parser, required=False)
    parser.add_argument("--figure", type=Path, default=Path("output/molecule_xsection_300K_1bar.png"))
    return parser


def main_xsection() -> None:
    args = build_xsection_parser().parse_args()
    run_xsection(args)


def run_xsection(args: argparse.Namespace) -> None:
    _, _, spectrum, line_provider = _compute_requested_absorption_spectrum(args)
    plot_absorption_spectrum(spectrum, args.figure)
    print(f"Broadening: {line_provider.broadening_summary()}")
    print(f"Species: {spectrum.species_name}")
    print(f"Grid points: {spectrum.wavenumber_cm1.size}")
    _print_positive_summary(spectrum.sigma_line_cm2_molecule, "Line cross section", "cm^2 / molecule")
    _print_positive_summary(spectrum.sigma_cia_cm2_molecule, "CIA / Continuum cross section", "cm^2 / molecule")
    _print_positive_summary(spectrum.sigma_total_cm2_molecule, "Total cross section", "cm^2 / molecule")


def build_attenuation_parser(*, require_external_cia: bool = False, description: str | None = None, default_figure: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or "Compute and plot molecular attenuation coefficient."
    )
    _add_common_arguments(parser)
    _add_external_cia_arguments(parser, required=require_external_cia)
    parser.add_argument(
        "--figure",
        type=Path,
        default=default_figure or Path("output/molecule_attenuation_300K_1bar.png"),
    )
    return parser


def main_attenuation() -> None:
    args = build_attenuation_parser().parse_args()
    run_attenuation(args)


def run_attenuation(args: argparse.Namespace) -> None:
    _, _, spectrum, line_provider = _compute_requested_absorption_spectrum(args)
    plot_attenuation_spectrum(spectrum, args.figure)
    print(f"Broadening: {line_provider.broadening_summary()}")
    _print_spectrum_summary(spectrum, include_components=True)


def main_attenuation_with_cia() -> None:
    args = build_attenuation_parser(
        require_external_cia=True,
        description="Compute and plot molecular line, CIA, and total attenuation on one graph.",
        default_figure=Path("output/molecule_plus_cia_attenuation_300K_1bar.png"),
    ).parse_args()
    run_attenuation(args)


def build_transmission_parser(*, require_external_cia: bool = False, description: str | None = None, default_figure: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or "Compute and plot molecular transmission over a fixed path length."
    )
    _add_common_arguments(parser)
    _add_external_cia_arguments(parser, required=require_external_cia)
    parser.add_argument("--path-length-km", type=float, default=1.0)
    parser.add_argument(
        "--figure",
        type=Path,
        default=default_figure or Path("output/molecule_transmission_300K_1bar_1km.png"),
    )
    return parser


def _compute_requested_transmittance(args: argparse.Namespace):
    if args.path_length_km <= 0.0:
        raise ValueError("path-length-km must be positive")
    _, _, spectrum, line_provider = _compute_requested_absorption_spectrum(args)
    transmittance = compute_transmittance_spectrum(
        spectrum=spectrum,
        path_length_m=float(args.path_length_km) * 1000.0,
    )
    return spectrum, transmittance, line_provider


def main_transmission() -> None:
    args = build_transmission_parser().parse_args()
    run_transmission(args)


def run_transmission(args: argparse.Namespace) -> None:
    _, transmittance, line_provider = _compute_requested_transmittance(args)
    plot_transmittance_spectrum(transmittance, args.figure)
    print(f"Broadening: {line_provider.broadening_summary()}")
    _print_transmittance_summary(transmittance, include_components=True)


def main_transmission_with_cia() -> None:
    args = build_transmission_parser(
        require_external_cia=True,
        description="Compute and plot molecular line, CIA, and total transmission on one graph.",
        default_figure=Path("output/molecule_plus_cia_transmission_300K_1bar_1km.png"),
    ).parse_args()
    run_transmission(args)


def build_line_positions_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and plot molecular HITRAN line positions and strengths.")
    _add_common_arguments(parser, include_state=False)
    parser.add_argument("--figure", type=Path, default=Path("output/molecule_lines.png"))
    return parser


def main_line_positions() -> None:
    args = build_line_positions_parser().parse_args()
    run_line_positions(args)


def run_line_positions(args: argparse.Namespace) -> None:
    band, config = _build_band_and_config(args)
    line_list = load_hitran_line_list(config, band)
    plot_hitran_line_positions(
        line_list,
        args.figure,
        wavenumber_min_cm1=band.wavenumber_min_cm1,
        wavenumber_max_cm1=band.wavenumber_max_cm1,
        min_line_strength=config.min_line_strength,
    )
    print(f"Species: {line_list.species_name}")
    print(f"Table: {line_list.table_name}")
    print(f"Lines: {line_list.wavenumber_cm1.size}")
    _print_positive_summary(line_list.line_intensity, "Line strength", "cm^-1 / (molecule cm^-2)")


def _mask_nonpositive(values: np.ndarray) -> np.ndarray:
    masked = np.asarray(values, dtype=np.float64).copy()
    masked[masked <= 0.0] = np.nan
    return masked


def _apply_positive_log_scale(ax, series: list[np.ndarray]) -> bool:
    has_positive = any(np.any(np.asarray(values, dtype=np.float64) > 0.0) for values in series)
    if has_positive:
        ax.set_yscale("log")
    return has_positive


def _add_legend_if_needed(ax) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)


def _style_shared_axis(ax, *, xlim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(labelsize=8)


def _resolve_overview_component_label(*, cia_dataset, spectrum) -> str:
    if cia_dataset is not None:
        return "CIA"
    if np.any(np.asarray(spectrum.sigma_cia_cm2_molecule, dtype=np.float64) > 0.0):
        return "Continuum"
    return "Secondary"


def _resolve_overview_component_title(*, cia_dataset, spectrum) -> str:
    if cia_dataset is not None:
        return f"CIA binary absorption coefficient ({cia_dataset.pair})"
    if spectrum.species_name.upper() == "H2O":
        return "H2O continuum contribution (MT_CKD)"
    return f"{spectrum.species_name} continuum contribution"


def _make_overview_header_text(*, spectrum, transmittance, band) -> str:
    return (
        f"{spectrum.species_name} overview | "
        f"{band.wavenumber_min_cm1:.3f}-{band.wavenumber_max_cm1:.3f} cm$^{{-1}}$ | "
        f"T={spectrum.temperature_k:.1f} K | p={spectrum.pressure_pa / 1.0e5:.3f} bar | "
        f"L={transmittance.path_length_m / 1000.0:.3f} km"
    )


def _plot_line_positions_panel(
    ax,
    line_list,
    *,
    xlim: tuple[float, float],
    min_line_strength: float = 1.0e-27,
) -> None:
    mask = (
        (line_list.wavenumber_cm1 >= xlim[0])
        & (line_list.wavenumber_cm1 <= xlim[1])
        & (line_list.line_intensity >= float(min_line_strength))
    )
    wavenumber = line_list.wavenumber_cm1[mask]
    strength = line_list.line_intensity[mask]
    if wavenumber.size == 0:
        ax.text(0.5, 0.5, "No line strengths above the minimum threshold in range.", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Line strength", fontsize=9)
        _style_shared_axis(ax, xlim=xlim)
        return
    ymin = float(min_line_strength)
    ax.vlines(wavenumber, ymin, strength, color="0.25", alpha=0.2, linewidth=0.35)
    ax.scatter(wavenumber, strength, s=5.0, color="tab:blue", alpha=0.75, linewidths=0.0)
    ax.set_yscale("log")
    ax.set_ylabel("Line strength\n[cm$^{-1}$/ (molecule cm$^{-2}$)]", fontsize=9)
    ax.set_title("Molecular line positions and strengths", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)


def _plot_cross_section_panel(ax, spectrum, *, xlim: tuple[float, float], show_components: bool, component_label: str) -> None:
    if show_components:
        ax.plot(spectrum.wavenumber_cm1, _mask_nonpositive(spectrum.sigma_line_cm2_molecule), label="Line", linewidth=1.0)
        ax.plot(
            spectrum.wavenumber_cm1,
            _mask_nonpositive(spectrum.sigma_cia_cm2_molecule),
            label=component_label,
            linewidth=1.0,
        )
    ax.plot(
        spectrum.wavenumber_cm1,
        _mask_nonpositive(spectrum.sigma_total_cm2_molecule),
        label="Total" if show_components else "Cross section",
        linewidth=1.4,
    )
    if not _apply_positive_log_scale(
        ax,
        [
            spectrum.sigma_total_cm2_molecule,
            spectrum.sigma_line_cm2_molecule if show_components else np.zeros(1, dtype=np.float64),
            spectrum.sigma_cia_cm2_molecule if show_components else np.zeros(1, dtype=np.float64),
        ],
    ):
        ax.text(0.5, 0.5, "No positive cross-section values.", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Cross section\n[cm$^2$ / molecule]", fontsize=9)
    ax.set_title("Absorption cross section", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)
    _add_legend_if_needed(ax)


def _plot_secondary_component_panel(ax, *, cia_dataset, spectrum, temperature_k: float, xlim: tuple[float, float]) -> None:
    if cia_dataset is not None:
        values = cia_dataset.interpolate_to_grid(temperature_k, spectrum.wavenumber_cm1)
        ylabel = "CIA binary coeff.\n[cm$^5$ / molecule$^2$]"
        empty_text = "No positive CIA values on the requested grid."
    else:
        values = spectrum.sigma_cia_cm2_molecule
        ylabel = "Continuum cross section\n[cm$^2$ / molecule]"
        empty_text = "No positive continuum contribution on the requested grid."
    ax.plot(spectrum.wavenumber_cm1, _mask_nonpositive(values), color="tab:purple", linewidth=1.2)
    if not _apply_positive_log_scale(ax, [values]):
        ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(_resolve_overview_component_title(cia_dataset=cia_dataset, spectrum=spectrum), fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)


def _plot_attenuation_panel(ax, spectrum, *, xlim: tuple[float, float], show_components: bool, component_label: str) -> None:
    if show_components:
        ax.plot(spectrum.wavenumber_cm1, _mask_nonpositive(spectrum.attenuation_line_m1), label="Line", linewidth=1.0)
        ax.plot(
            spectrum.wavenumber_cm1,
            _mask_nonpositive(spectrum.attenuation_cia_m1),
            label=component_label,
            linewidth=1.0,
        )
    ax.plot(
        spectrum.wavenumber_cm1,
        _mask_nonpositive(spectrum.attenuation_total_m1),
        label="Total" if show_components else "Attenuation",
        linewidth=1.4,
    )
    if not _apply_positive_log_scale(
        ax,
        [
            spectrum.attenuation_total_m1,
            spectrum.attenuation_line_m1 if show_components else np.zeros(1, dtype=np.float64),
            spectrum.attenuation_cia_m1 if show_components else np.zeros(1, dtype=np.float64),
        ],
    ):
        ax.text(0.5, 0.5, "No positive attenuation values.", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Attenuation\n[1 / m]", fontsize=9)
    ax.set_title("Attenuation coefficient", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)
    _add_legend_if_needed(ax)


def _plot_transmittance_panel(ax, transmittance, *, xlim: tuple[float, float], show_components: bool, component_label: str) -> None:
    if show_components:
        ax.plot(transmittance.wavenumber_cm1, transmittance.transmittance_line, label="Line", linewidth=1.0)
        ax.plot(
            transmittance.wavenumber_cm1,
            transmittance.transmittance_cia,
            label=component_label,
            linewidth=1.0,
        )
    ax.plot(
        transmittance.wavenumber_cm1,
        transmittance.transmittance_total,
        label="Total" if show_components else "Transmittance",
        linewidth=1.4,
    )
    ax.plot(
        transmittance.wavenumber_cm1,
        compute_normalized_blackbody_curve(
            wavenumber_cm1=transmittance.wavenumber_cm1,
            temperature_k=transmittance.temperature_k,
        ),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Blackbody",
    )
    ax.set_ylabel("Transmittance", fontsize=9)
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Transmittance", fontsize=10, loc="left")
    _style_shared_axis(ax, xlim=xlim)
    _add_legend_if_needed(ax)


def _render_overview_page(fig, axes_flat, *, band, config, spectrum, transmittance, line_list, cia_dataset) -> None:
    xlim = (band.wavenumber_min_cm1, band.wavenumber_max_cm1)
    fig.subplots_adjust(
        left=1.0 / 8.5,
        right=1.0 - 1.0 / 8.5,
        bottom=1.0 / 11.0,
        top=1.0 - 1.0 / 11.0,
        hspace=0.5,
    )
    fig.text(
        1.0 / 8.5,
        1.0 - 0.45 / 11.0,
        _make_overview_header_text(spectrum=spectrum, transmittance=transmittance, band=band),
        ha="left",
        va="center",
        fontsize=11,
    )

    has_secondary_component = bool(np.any(np.asarray(spectrum.sigma_cia_cm2_molecule, dtype=np.float64) > 0.0))
    component_label = _resolve_overview_component_label(cia_dataset=cia_dataset, spectrum=spectrum)
    show_components = has_secondary_component
    _plot_line_positions_panel(
        axes_flat[0],
        line_list,
        xlim=xlim,
        min_line_strength=config.min_line_strength,
    )
    _plot_cross_section_panel(axes_flat[1], spectrum, xlim=xlim, show_components=show_components, component_label=component_label)
    if show_components:
        _plot_secondary_component_panel(
            axes_flat[2],
            cia_dataset=cia_dataset,
            spectrum=spectrum,
            temperature_k=spectrum.temperature_k,
            xlim=xlim,
        )
        _plot_attenuation_panel(axes_flat[3], spectrum, xlim=xlim, show_components=True, component_label=component_label)
        _plot_transmittance_panel(axes_flat[4], transmittance, xlim=xlim, show_components=True, component_label=component_label)
        last_data_axis_index = 4
    else:
        _plot_attenuation_panel(axes_flat[2], spectrum, xlim=xlim, show_components=False, component_label=component_label)
        _plot_transmittance_panel(axes_flat[3], transmittance, xlim=xlim, show_components=False, component_label=component_label)
        axes_flat[4].axis("off")
        last_data_axis_index = 3

    for idx, ax in enumerate(axes_flat):
        if idx <= last_data_axis_index:
            ax.set_xlabel("Wavenumber [cm$^{-1}$]", fontsize=10)


def _compute_overview_products(args: argparse.Namespace):
    if args.path_length_km <= 0.0:
        raise ValueError("path-length-km must be positive")
    band, config = _build_band_and_config(args)
    line_db = download_hitran_lines(config, band)
    line_list = load_hitran_line_list(config, band, line_db=line_db)
    cia_dataset = _load_requested_cia_dataset(args, config)
    _, _, spectrum, line_provider = _compute_requested_absorption_spectrum(args, line_db=line_db, cia_dataset=cia_dataset)
    transmittance = compute_transmittance_spectrum(
        spectrum=spectrum,
        path_length_m=float(args.path_length_km) * 1000.0,
    )
    return band, config, line_list, spectrum, transmittance, cia_dataset, line_provider


def _print_overview_summary(*, figure_path: Path, spectrum, transmittance, cia_dataset, line_provider) -> None:
    print(f"Overview figure: {figure_path}")
    print(f"Broadening: {line_provider.broadening_summary()}")
    print(f"Species: {spectrum.species_name}")
    print(f"Grid points: {spectrum.wavenumber_cm1.size}")
    print(f"Path length: {transmittance.path_length_m / 1000.0:.3f} km")
    if cia_dataset is not None:
        print(f"CIA file: {cia_dataset.source_path}")
        print(f"CIA pair: {cia_dataset.pair}")


class _SplitSpeciesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        species: list[str] = []
        for value in values:
            species.extend(part.strip() for part in str(value).split(",") if part.strip())
        setattr(namespace, self.dest, species)


def build_molecule_overview_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a combined multi-page PDF of molecule overview plots over one or more wavenumber ranges."
    )
    parser.add_argument("--hitran-dir", type=Path, default=Path("hitran"))
    parser.add_argument("--species", nargs="+", action=_SplitSpeciesAction, default=["H2", "CO2", "H2O", "CH4", "N2"])
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--path-length-km", type=float, default=1.0)
    parser.add_argument("--wn-range", dest="wn_ranges", action="append", type=_parse_wn_range, required=True)
    parser.add_argument("--refresh-hitran", action="store_true")
    parser.add_argument("--broadening-composition", default=None)
    parser.add_argument("--cia-index-url", default="https://hitran.org/cia/")
    parser.add_argument("--refresh-cia", action="store_true")
    parser.add_argument("--figure", type=Path, default=Path("output/molecule_overview_collection.pdf"))
    return parser


def main_overview_batch() -> None:
    args = build_molecule_overview_batch_parser().parse_args()
    run_overview_batch(args)


def run_overview_batch(args: argparse.Namespace) -> None:
    if args.path_length_km <= 0.0:
        raise ValueError("path-length-km must be positive")
    figure_path = args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    page_tasks: list[argparse.Namespace] = []
    page_metadata: list[tuple[float, float, str, float, float]] = []
    for temperature_k, pressure_bar in _state_pairs(args):
        for species in args.species:
            for wn_min, wn_max in args.wn_ranges:
                page_args = argparse.Namespace(
                    hitran_dir=args.hitran_dir,
                    species=species,
                    temperature_k=float(temperature_k),
                    pressure_bar=float(pressure_bar),
                    wn_range=(wn_min, wn_max),
                    resolution=args.resolution,
                    refresh_hitran=args.refresh_hitran,
                    broadening_composition=args.broadening_composition,
                    cia_filename=None,
                    cia_pair=None,
                    cia_index_url=args.cia_index_url,
                    refresh_cia=args.refresh_cia,
                    path_length_km=args.path_length_km,
                    figure=figure_path,
                )
                page_tasks.append(page_args)
                page_metadata.append((float(temperature_k), float(pressure_bar), species, wn_min, wn_max))
    page_results = _parallel_overview_page_products(page_tasks)
    page_count = 0
    with PdfPages(figure_path) as pdf:
        for (temperature_k, pressure_bar, species, wn_min, wn_max), result in zip(page_metadata, page_results, strict=True):
            band, config, line_list, spectrum, transmittance, cia_dataset, _ = result
            fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8.5, 11.0), squeeze=False)
            _render_overview_page(
                fig,
                axes[:, 0],
                band=band,
                config=config,
                spectrum=spectrum,
                transmittance=transmittance,
                line_list=line_list,
                cia_dataset=cia_dataset,
            )
            pdf.savefig(fig)
            plt.close(fig)
            page_count += 1
            component_label = _resolve_overview_component_label(cia_dataset=cia_dataset, spectrum=spectrum) if np.any(np.asarray(spectrum.sigma_cia_cm2_molecule, dtype=np.float64) > 0.0) else "none"
            print(
                f"Added page {page_count}: T={temperature_k:.1f} K | p={pressure_bar:.3f} bar | {species} | {wn_min:.3f}-{wn_max:.3f} cm^-1 | secondary={component_label}"
            )
    print(f"Combined overview figure: {figure_path}")
    print(f"Pages: {page_count}")


def build_overview_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a US-letter PDF overview of molecular lines, cross section, optional CIA, attenuation, and transmittance."
    )
    _add_common_arguments(parser)
    _add_external_cia_arguments(parser, required=False)
    parser.add_argument("--path-length-km", type=float, default=1.0)
    parser.add_argument("--figure", type=Path, default=Path("output/molecule_overview_300K_1bar.pdf"))
    return parser


def main_overview() -> None:
    args = build_overview_parser().parse_args()
    run_overview(args)


def run_overview(args: argparse.Namespace) -> None:
    figure_path = args.figure
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    page_tasks: list[argparse.Namespace] = []
    for temperature_k, pressure_bar in _state_pairs(args):
        page_args = argparse.Namespace(**vars(args))
        page_args.temperature_k = float(temperature_k)
        page_args.pressure_bar = float(pressure_bar)
        page_tasks.append(page_args)
    page_results = _parallel_overview_page_products(page_tasks)

    with PdfPages(figure_path) as pdf:
        for temperature_k, pressure_bar, result in zip(_selected_temperatures(args), _selected_pressure_bars(args), page_results, strict=True):
            band, config, line_list, spectrum, transmittance, cia_dataset, broadening_summary = result
            fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8.5, 11.0), squeeze=False)
            _render_overview_page(
                fig,
                axes[:, 0],
                band=band,
                config=config,
                spectrum=spectrum,
                transmittance=transmittance,
                line_list=line_list,
                cia_dataset=cia_dataset,
            )
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Added page: T={temperature_k:.1f} K | p={pressure_bar:.3f} bar")

    band, config, line_list, spectrum, transmittance, cia_dataset, broadening_summary = page_results[-1]
    _print_overview_summary(
        figure_path=figure_path,
        spectrum=spectrum,
        transmittance=transmittance,
        cia_dataset=cia_dataset,
        line_provider=type("_Summary", (), {"broadening_summary": lambda self: broadening_summary})(),
    )


def _compute_overview_page_task(args: argparse.Namespace):
    band, config, line_list, spectrum, transmittance, cia_dataset, line_provider = _compute_overview_products(args)
    return band, config, line_list, spectrum, transmittance, cia_dataset, line_provider.broadening_summary()


def _parallel_overview_page_products(tasks: list[argparse.Namespace]):
    if len(tasks) <= 1:
        return [_compute_overview_page_task(task) for task in tasks]
    max_workers = min(len(tasks), os.cpu_count() or 1)
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        return list(executor.map(_compute_overview_page_task, tasks))


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
