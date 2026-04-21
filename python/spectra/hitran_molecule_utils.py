"""Utility helpers for HITRAN molecular line workflows."""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectra_matplotlib"))

from .config import SpectralBandConfig, SpectroscopyConfig, parse_broadening_composition, resolve_hitran_cia_pair
from .hitran_cia_utils import load_cia_dataset
from .utils import build_band_from_range


def _import_hapi():
    try:
        return importlib.import_module("hapi")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("HAPI is required for HITRAN line downloads.") from exc


def _resolve_continuum_sources(**kwargs):
    from .spectrum import _resolve_continuum_sources as resolve_continuum_sources

    return resolve_continuum_sources(**kwargs)


def compute_absorption_spectrum_from_sources(**kwargs):
    from .spectrum import compute_absorption_spectrum_from_sources as compute_from_sources

    return compute_from_sources(**kwargs)


def compute_transmittance_spectrum(**kwargs):
    from .transmittance import compute_transmittance_spectrum as compute_transmittance

    return compute_transmittance(**kwargs)


@dataclass(frozen=True)
class LineDatabase:
    """Description of the cached HITRAN line table."""

    table_name: str
    cache_dir: Path
    wavenumber_min_cm1: float
    wavenumber_max_cm1: float
    available_broadener_keys: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HitranLineList:
    """Discrete HITRAN line positions and tabulated intensities."""

    species_name: str
    table_name: str
    wavenumber_cm1: np.ndarray
    line_intensity: np.ndarray


class HapiLineProvider:
    """Compute line absorption coefficients from a cached HAPI table."""

    def __init__(
        self,
        table_name: str,
        cache_dir: Path | None = None,
        diluent: dict[str, float] | None = None,
        min_line_strength: float = 1.0e-27,
        available_broadener_keys: tuple[str, ...] | None = None,
    ) -> None:
        self._hapi = _import_hapi()
        self.table_name = table_name
        self.cache_dir = cache_dir
        requested_diluent = dict(diluent or {"self": 1.0})
        self.requested_diluent = requested_diluent
        self.diluent, self.diluent_fallbacks = _resolve_effective_diluent(
            self._hapi,
            table_name=table_name,
            cache_dir=cache_dir,
            requested_diluent=requested_diluent,
            available=available_broadener_keys,
        )
        self.min_line_strength = float(min_line_strength)
        if cache_dir is not None:
            _call_hapi_quietly(self._hapi.db_begin, str(cache_dir))

    def broadening_summary(self) -> str:
        """Return a compact description of requested and effective broadening."""
        requested = _format_diluent(self.requested_diluent)
        effective = _format_diluent(self.diluent)
        if not self.diluent_fallbacks:
            return f"requested={requested} -> effective={effective}"
        fallback_text = ", ".join(f"{name}->{target}" for name, target in sorted(self.diluent_fallbacks.items()))
        return f"requested={requested} -> effective={effective} (fallback: {fallback_text})"

    def absorption_coefficient_cm1(
        self,
        wavenumber_grid_cm1: np.ndarray,
        temperature_k: float,
        pressure_pa: float,
    ) -> np.ndarray:
        """Return line absorption coefficient on the requested grid."""
        pressure_atm = float(pressure_pa) / 101_325.0
        _, coef = _call_hapi_quietly(
            self._hapi.absorptionCoefficient_Voigt,
            SourceTables=self.table_name,
            OmegaGrid=np.asarray(wavenumber_grid_cm1, dtype=np.float64),
            Environment={"T": float(temperature_k), "p": pressure_atm},
            Diluent=self.diluent,
            IntensityThreshold=self.min_line_strength,
            HITRAN_units=False,
        )
        return np.asarray(coef, dtype=np.float64)

    def cross_section_cm2_molecule(
        self,
        wavenumber_grid_cm1: np.ndarray,
        temperature_k: float,
        pressure_pa: float,
    ) -> np.ndarray:
        """Return line absorption cross section in cm^2/molecule."""
        pressure_atm = float(pressure_pa) / 101_325.0
        _, coef = _call_hapi_quietly(
            self._hapi.absorptionCoefficient_Voigt,
            SourceTables=self.table_name,
            OmegaGrid=np.asarray(wavenumber_grid_cm1, dtype=np.float64),
            Environment={"T": float(temperature_k), "p": pressure_atm},
            Diluent=self.diluent,
            IntensityThreshold=self.min_line_strength,
            HITRAN_units=True,
        )
        return np.asarray(coef, dtype=np.float64)


def _resolve_global_isotopologue_ids(hapi, config: SpectroscopyConfig) -> tuple[int, ...]:
    """Translate molecule-local isotope numbers to HITRAN global isotope ids."""
    iso_index = hapi.ISO_INDEX["id"]
    global_ids: list[int] = []
    for local_iso_id in config.resolved_isotopologue_ids():
        try:
            global_ids.append(int(hapi.ISO[(config.molecule_id, int(local_iso_id))][iso_index]))
        except KeyError as exc:
            raise ValueError(
                f"Unsupported isotopologue {(config.molecule_id, int(local_iso_id))} "
                "for the configured molecule."
            ) from exc
    return tuple(global_ids)


def _call_hapi_quietly(func, *args, **kwargs):
    return func(*args, **kwargs)


def _cache_matches_requested_molecule_from_data(data: dict[str, object], molecule_id: int) -> bool:
    try:
        molec_ids = data["molec_id"]
    except Exception:
        return False
    return {int(value) for value in molec_ids} == {int(molecule_id)}


def _available_broadener_keys_from_data(data: dict[str, object]) -> set[str]:
    return {
        str(key)[len("gamma_") :].lower()
        for key in data
        if str(key).startswith("gamma_")
    }


def _load_cached_table_data(hapi, table_name: str, cache_dir: Path | None) -> dict[str, object] | None:
    if cache_dir is not None:
        _call_hapi_quietly(hapi.db_begin, str(cache_dir))
    try:
        _call_hapi_quietly(hapi.storage2cache, table_name)
        return hapi.LOCAL_TABLE_CACHE[table_name]["data"]
    except Exception:
        return None


def _resolve_effective_diluent(
    hapi,
    *,
    table_name: str,
    cache_dir: Path | None,
    requested_diluent: dict[str, float],
    available: tuple[str, ...] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    available_keys = set(available) if available is not None else _available_broadener_keys_from_data(_load_cached_table_data(hapi, table_name, cache_dir) or {})
    effective: dict[str, float] = {}
    fallbacks: dict[str, str] = {}
    needs_air_fallback = False
    for broadener_name, fraction in requested_diluent.items():
        key = str(broadener_name).strip().lower()
        amount = float(fraction)
        if amount <= 0.0:
            continue
        if key == "self" or key in available_keys:
            effective[key] = effective.get(key, 0.0) + amount
            continue
        needs_air_fallback = True
        fallbacks[key] = "air"
        effective["air"] = effective.get("air", 0.0) + amount
    if needs_air_fallback and "air" not in available_keys:
        missing = ", ".join(sorted(name for name, target in fallbacks.items() if target == "air"))
        raise ValueError(
            f"Requested broadeners {missing} are unavailable for table {table_name}, "
            "and air broadening parameters are not available for fallback."
        )
    if not effective:
        return {"self": 1.0}, {}
    total = sum(effective.values())
    return ({name: value / total for name, value in effective.items()}, fallbacks)


def _format_diluent(diluent: dict[str, float]) -> str:
    return ",".join(f"{name}:{value:.3f}" for name, value in sorted(diluent.items()))


def download_hitran_lines(config: SpectroscopyConfig, band: SpectralBandConfig) -> LineDatabase:
    """Fetch the configured species line table for one spectral band."""
    config.ensure_directories()
    hapi = _import_hapi()
    bounds_min = band.wavenumber_min_cm1
    bounds_max = band.wavenumber_max_cm1
    table_name = config.resolved_line_table_name(band)
    global_iso_ids = _resolve_global_isotopologue_ids(hapi, config)
    _call_hapi_quietly(hapi.db_begin, str(config.hitran_cache_dir))
    data_path = config.hitran_cache_dir / f"{table_name}.data"
    header_path = config.hitran_cache_dir / f"{table_name}.header"
    cached_data = None
    available_broadener_keys: tuple[str, ...] | None = None
    if data_path.exists() and header_path.exists():
        cached_data = _load_cached_table_data(hapi, table_name, config.hitran_cache_dir)
        if cached_data is not None:
            available_broadener_keys = tuple(sorted(_available_broadener_keys_from_data(cached_data)))
    cache_is_valid = (
        data_path.exists()
        and header_path.exists()
        and cached_data is not None
        and _cache_matches_requested_molecule_from_data(cached_data, config.molecule_id)
    )
    if config.refresh_hitran or not cache_is_valid:
        previous_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(30.0)
            _call_hapi_quietly(
                hapi.fetch_by_ids,
                table_name,
                list(global_iso_ids),
                bounds_min,
                bounds_max,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download HITRAN lines for {config.hitran_species.name} over "
                f"{bounds_min:g}-{bounds_max:g} cm^-1 into {config.hitran_cache_dir}. "
                "The requested table is not available in the local cache and HAPI could not fetch it."
            ) from exc
        finally:
            socket.setdefaulttimeout(previous_timeout)
    return LineDatabase(
        table_name=table_name,
        cache_dir=config.hitran_cache_dir,
        wavenumber_min_cm1=bounds_min,
        wavenumber_max_cm1=bounds_max,
        available_broadener_keys=available_broadener_keys,
    )


def load_hitran_line_list(
    config: SpectroscopyConfig,
    band: SpectralBandConfig,
    line_db: LineDatabase | None = None,
) -> HitranLineList:
    """Load discrete HITRAN line positions and tabulated intensities for the configured species."""
    line_db = line_db or download_hitran_lines(config, band)
    hapi = _import_hapi()
    _call_hapi_quietly(hapi.db_begin, str(line_db.cache_dir))
    _call_hapi_quietly(hapi.storage2cache, line_db.table_name)
    data = hapi.LOCAL_TABLE_CACHE[line_db.table_name]["data"]
    wavenumber_cm1 = np.asarray(data["nu"], dtype=np.float64)
    line_intensity = np.asarray(data["sw"], dtype=np.float64)
    valid = (
        np.isfinite(wavenumber_cm1)
        & np.isfinite(line_intensity)
        & (wavenumber_cm1 > 0.0)
        & (line_intensity >= float(config.min_line_strength))
    )
    order = np.argsort(wavenumber_cm1[valid])
    return HitranLineList(
        species_name=config.hitran_species.name,
        table_name=line_db.table_name,
        wavenumber_cm1=wavenumber_cm1[valid][order],
        line_intensity=line_intensity[valid][order],
    )


def build_line_provider(
    config: SpectroscopyConfig,
    line_db: LineDatabase,
) -> HapiLineProvider:
    """Construct a HAPI line provider using the config's broadening rules."""
    return HapiLineProvider(
        line_db.table_name,
        cache_dir=line_db.cache_dir,
        diluent=config.resolved_line_diluent(),
        min_line_strength=config.min_line_strength,
        available_broadener_keys=line_db.available_broadener_keys,
    )


def _build_band_and_config(args: argparse.Namespace) -> tuple[SpectralBandConfig, SpectroscopyConfig]:
    band = build_band_from_range(tuple(args.wn_range), float(args.resolution))
    config = SpectroscopyConfig(
        output_path=Path("output") / "unused.nc",
        hitran_cache_dir=args.hitran_dir,
        species_name=args.species,
        broadening_composition=parse_broadening_composition(getattr(args, "broadening_composition", None)),
        refresh_hitran=args.refresh_hitran,
    )
    return band, config


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


def load_requested_cia_dataset(args: argparse.Namespace, config: SpectroscopyConfig):
    cia_filename = _resolve_self_component_filename(args, config)
    if cia_filename is None:
        return None
    return load_cia_dataset(
        cache_dir=args.hitran_dir,
        filename=cia_filename,
        pair=_resolve_cia_pair(args, config),
        refresh=bool(args.refresh_cia),
    )


_MISSING = object()


def compute_requested_absorption_spectrum(
    args: argparse.Namespace,
    *,
    line_db: LineDatabase | None = None,
    cia_dataset=_MISSING,
):
    band, config = _build_band_and_config(args)
    temperature_k = float(args.temperature_k)
    pressure_pa = float(args.pressure_bar) * 1.0e5
    if cia_dataset is _MISSING:
        cia_dataset = load_requested_cia_dataset(args, config)
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


def compute_requested_transmittance(args: argparse.Namespace):
    if args.path_length_km <= 0.0:
        raise ValueError("path-length-km must be positive")
    _, _, spectrum, line_provider = compute_requested_absorption_spectrum(args)
    transmittance = compute_transmittance_spectrum(
        spectrum=spectrum,
        path_length_m=float(args.path_length_km) * 1000.0,
    )
    return spectrum, transmittance, line_provider


def selected_temperatures(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "temperature_k", [300.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def selected_pressure_bars(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "pressure_bar", [1.0])
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def state_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    temperatures = selected_temperatures(args)
    pressures = selected_pressure_bars(args)
    if len(temperatures) != len(pressures):
        raise ValueError("temperature_k and pressure_bar must have the same number of values")
    return list(zip(temperatures, pressures, strict=True))
