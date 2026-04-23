"""Utility helpers for HITRAN collision-induced absorption data."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np

from .config import SpectroscopyConfig

K_BOLTZMANN = 1.380649e-23
HITRAN_CIA_INDEX_URL = "https://hitran.org/cia/"


class _HrefCollector(HTMLParser):
    """Collect links from an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class CiaBlock:
    """One temperature-dependent CIA block from a HITRAN CIA file."""

    temperature_k: float
    wavenumber_cm1: np.ndarray
    binary_cross_section_cm5_molecule2: np.ndarray


@dataclass(frozen=True)
class CiaDataset:
    """Temperature-dependent CIA data for one collisional pair."""

    pair: str
    blocks: tuple[CiaBlock, ...]
    source_path: Path

    @property
    def temperatures_k(self) -> np.ndarray:
        return np.array(sorted({block.temperature_k for block in self.blocks}), dtype=np.float64)

    def _spectra_by_temperature(
        self,
        wavenumber_grid_cm1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one merged spectrum per temperature on the target grid."""
        temperatures_k = self.temperatures_k
        spectra = np.full((temperatures_k.size, wavenumber_grid_cm1.size), np.nan, dtype=np.float64)
        for idx, temperature_k in enumerate(temperatures_k):
            matching_blocks = [block for block in self.blocks if block.temperature_k == temperature_k]
            for block in matching_blocks:
                block_values = np.interp(
                    wavenumber_grid_cm1,
                    block.wavenumber_cm1,
                    block.binary_cross_section_cm5_molecule2,
                    left=np.nan,
                    right=np.nan,
                )
                valid = ~np.isnan(block_values)
                spectra[idx, valid] = block_values[valid]
        return temperatures_k, spectra

    def interpolate_to_grid(
        self,
        temperature_k: float,
        wavenumber_grid_cm1: np.ndarray,
    ) -> np.ndarray:
        """Interpolate CIA binary cross section in temperature and wavenumber."""
        wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
        temperatures_k, spectra = self._spectra_by_temperature(wavenumber_grid_cm1)
        interpolated = np.zeros(wavenumber_grid_cm1.shape, dtype=np.float64)

        for wave_idx in range(wavenumber_grid_cm1.size):
            column = spectra[:, wave_idx]
            valid = ~np.isnan(column)
            if not np.any(valid):
                continue
            valid_temperatures = temperatures_k[valid]
            valid_values = column[valid]
            if temperature_k <= valid_temperatures[0]:
                interpolated[wave_idx] = valid_values[0]
                continue
            if temperature_k >= valid_temperatures[-1]:
                interpolated[wave_idx] = valid_values[-1]
                continue
            hi = int(np.searchsorted(valid_temperatures, temperature_k))
            lo = hi - 1
            weight = (temperature_k - valid_temperatures[lo]) / (valid_temperatures[hi] - valid_temperatures[lo])
            interpolated[wave_idx] = (1.0 - weight) * valid_values[lo] + weight * valid_values[hi]

        return interpolated


def _download_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "spectra/0.1"})
    with urlopen(req) as response:
        return response.read().decode("utf-8")


def find_cia_download_url(filename: str) -> str:
    """Resolve the CIA download URL by scraping the HITRAN CIA index page."""
    html = _download_text(HITRAN_CIA_INDEX_URL)
    parser = _HrefCollector()
    parser.feed(html)
    filename_lower = filename.lower()
    for href in parser.hrefs:
        if filename_lower in href.lower():
            return urljoin(HITRAN_CIA_INDEX_URL, href)
    return urljoin(HITRAN_CIA_INDEX_URL, filename)


def download_cia_file_by_name(
    *,
    cache_dir: Path,
    filename: str,
    refresh: bool = False,
) -> Path:
    """Download a named CIA file into the requested local cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.exists() and not refresh:
        return target
    url = find_cia_download_url(filename)
    req = Request(url, headers={"User-Agent": "spectra/0.1"})
    with urlopen(req) as response, open(target, "wb") as handle:
        handle.write(response.read())
    return target


def download_cia_file(config: SpectroscopyConfig) -> Path:
    """Download the configured CIA file into the local cache."""
    config.ensure_directories()
    return download_cia_file_by_name(
        cache_dir=config.hitran_cache_dir,
        filename=config.cia_filename,
        refresh=config.refresh_hitran,
    )


def parse_cia_file(path: Path, pair: str) -> CiaDataset:
    """Parse a HITRAN CIA file into temperature-tagged blocks."""
    blocks: list[CiaBlock] = []
    current_temperature: float | None = None
    current_wavenumber: list[float] = []
    current_cross_section: list[float] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            is_header = False
            if line.startswith("#"):
                is_header = True
                header_fields = line[1:].split()
            elif parts and "-" in parts[0] and len(parts) >= 5:
                is_header = True
                header_fields = parts
            else:
                header_fields = []

            if is_header:
                if current_temperature is not None and current_wavenumber:
                    blocks.append(
                        CiaBlock(
                            temperature_k=current_temperature,
                            wavenumber_cm1=np.array(current_wavenumber, dtype=np.float64),
                            binary_cross_section_cm5_molecule2=np.array(current_cross_section, dtype=np.float64),
                        )
                    )
                    current_wavenumber = []
                    current_cross_section = []
                if not header_fields:
                    continue
                if header_fields[0].upper() != pair.upper():
                    current_temperature = None
                    continue
                if len(header_fields) < 5:
                    raise ValueError(f"Malformed CIA header line: {line}")
                current_temperature = float(header_fields[4])
                continue

            if current_temperature is None:
                continue
            if len(parts) < 2:
                continue
            current_wavenumber.append(float(parts[0]))
            current_cross_section.append(float(parts[1]))

    if current_temperature is not None and current_wavenumber:
        blocks.append(
            CiaBlock(
                temperature_k=current_temperature,
                wavenumber_cm1=np.array(current_wavenumber, dtype=np.float64),
                binary_cross_section_cm5_molecule2=np.array(current_cross_section, dtype=np.float64),
            )
        )

    if not blocks:
        raise ValueError(f"No CIA data found for pair {pair!r} in {path}.")
    blocks.sort(key=lambda block: block.temperature_k)
    return CiaDataset(pair=pair, blocks=tuple(blocks), source_path=path)


def load_cia_dataset(
    *,
    cache_dir: Path,
    filename: str,
    pair: str,
    refresh: bool = False,
) -> CiaDataset:
    """Download and parse a CIA dataset identified by filename and pair."""
    path = download_cia_file_by_name(
        cache_dir=cache_dir,
        filename=filename,
        refresh=refresh,
    )
    return parse_cia_file(path, pair)


def compute_cia_attenuation_m1(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    pressure_pa: float,
    wavenumber_grid_cm1: np.ndarray,
) -> np.ndarray:
    """Convert a CIA binary coefficient into attenuation coefficient in 1/m."""
    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    binary_xsec = dataset.interpolate_to_grid(temperature_k, wavenumber_grid_cm1)
    number_density_cm3 = float(pressure_pa / (K_BOLTZMANN * temperature_k) / 1.0e6)
    return np.asarray(binary_xsec * number_density_cm3**2 * 100.0, dtype=np.float64)


def compute_cia_transmission(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    pressure_pa: float,
    path_length_m: float,
    wavenumber_grid_cm1: np.ndarray,
) -> np.ndarray:
    """Convert CIA attenuation into Beer-Lambert transmission over a fixed path length."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be positive")
    attenuation_m1 = compute_cia_attenuation_m1(
        dataset,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        wavenumber_grid_cm1=wavenumber_grid_cm1,
    )
    return np.exp(-attenuation_m1 * float(path_length_m))
