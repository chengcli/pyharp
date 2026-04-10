"""HITRAN/HAPI line-download and line-opacity helpers."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
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
    ) -> None:
        self._hapi = _import_hapi()
        self.table_name = table_name
        self.cache_dir = cache_dir
        self.diluent = diluent or {"self": 1.0}
        self.min_line_strength = float(min_line_strength)
        if cache_dir is not None:
            self._hapi.db_begin(str(cache_dir))

    def absorption_coefficient_cm1(
        self,
        wavenumber_grid_cm1: np.ndarray,
        temperature_k: float,
        pressure_pa: float,
    ) -> np.ndarray:
        """Return line absorption coefficient on the requested grid."""
        pressure_atm = float(pressure_pa) / 101_325.0
        _, coef = self._hapi.absorptionCoefficient_Voigt(
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
        _, coef = self._hapi.absorptionCoefficient_Voigt(
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


def _cache_matches_requested_molecule(hapi, table_name: str, molecule_id: int) -> bool:
    """Return True when the cached HAPI table contains only the requested molecule."""
    try:
        hapi.storage2cache(table_name)
    except Exception:
        return False
    try:
        molec_ids = hapi.LOCAL_TABLE_CACHE[table_name]["data"]["molec_id"]
    except Exception:
        return False
    return {int(value) for value in molec_ids} == {int(molecule_id)}


def download_hitran_lines(config: SpectroscopyConfig, band: SpectralBandConfig) -> LineDatabase:
    """Fetch the configured species line table for one spectral band."""
    config.ensure_directories()
    hapi = _import_hapi()
    bounds_min = band.wavenumber_min_cm1
    bounds_max = band.wavenumber_max_cm1
    table_name = config.resolved_line_table_name(band)
    global_iso_ids = _resolve_global_isotopologue_ids(hapi, config)
    hapi.db_begin(str(config.hitran_cache_dir))
    data_path = config.hitran_cache_dir / f"{table_name}.data"
    header_path = config.hitran_cache_dir / f"{table_name}.header"
    cache_is_valid = (
        data_path.exists()
        and header_path.exists()
        and _cache_matches_requested_molecule(hapi, table_name, config.molecule_id)
    )
    if config.refresh_hitran or not cache_is_valid:
        hapi.fetch_by_ids(
            table_name,
            list(global_iso_ids),
            bounds_min,
            bounds_max,
        )
    return LineDatabase(
        table_name=table_name,
        cache_dir=config.hitran_cache_dir,
        wavenumber_min_cm1=bounds_min,
        wavenumber_max_cm1=bounds_max,
    )


def load_hitran_line_list(
    config: SpectroscopyConfig,
    band: SpectralBandConfig,
    line_db: LineDatabase | None = None,
) -> HitranLineList:
    """Load discrete HITRAN line positions and tabulated intensities for the configured species."""
    line_db = line_db or download_hitran_lines(config, band)
    hapi = _import_hapi()
    hapi.db_begin(str(line_db.cache_dir))
    hapi.storage2cache(line_db.table_name)
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
