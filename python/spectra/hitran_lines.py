"""HITRAN/HAPI line-download and line-opacity helpers."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectra_matplotlib"))

from .config import SpectralBandConfig, SpectroscopyConfig


def _import_hapi():
    try:
        return importlib.import_module("hapi")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("HAPI is required for HITRAN line downloads.") from exc


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


def plot_hitran_line_positions(
    line_list: HitranLineList,
    figure_path: Path,
    *,
    wavenumber_min_cm1: float | None = None,
    wavenumber_max_cm1: float | None = None,
    min_line_strength: float = 1.0e-27,
) -> None:
    """Plot discrete HITRAN line positions and line intensities on log-log axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = float(min_line_strength)
    positive = line_list.line_intensity[line_list.line_intensity >= threshold]
    if positive.size == 0:
        raise ValueError("Line list does not contain any intensities above the minimum threshold.")
    ymin = threshold

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.vlines(line_list.wavenumber_cm1, ymin, line_list.line_intensity, color="0.2", alpha=0.2, linewidth=0.35)
    ax.scatter(line_list.wavenumber_cm1, line_list.line_intensity, s=4.0, color="tab:blue", alpha=0.7, linewidths=0.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ymin, max(float(np.max(positive)) * 1.2, ymin * 10.0))
    if wavenumber_min_cm1 is not None and wavenumber_max_cm1 is not None:
        ax.set_xlim(float(wavenumber_min_cm1), float(wavenumber_max_cm1))
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Line intensity $S$ [cm$^{-1}$/ (molecule cm$^{-2}$)]")
    ax.set_title(f"{line_list.species_name} HITRAN line positions and intensities")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
