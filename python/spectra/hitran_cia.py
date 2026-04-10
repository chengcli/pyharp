"""Download and parse HITRAN collision-induced absorption files."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np

from .blackbody import compute_normalized_blackbody_curve
from .config import SpectroscopyConfig

K_BOLTZMANN = 1.380649e-23


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


def find_cia_download_url(index_url: str, filename: str) -> str:
    """Resolve the CIA download URL by scraping the HITRAN CIA index page."""
    html = _download_text(index_url)
    parser = _HrefCollector()
    parser.feed(html)
    filename_lower = filename.lower()
    for href in parser.hrefs:
        if filename_lower in href.lower():
            return urljoin(index_url, href)
    return urljoin(index_url, filename)


def download_cia_file_by_name(
    *,
    cache_dir: Path,
    filename: str,
    index_url: str = "https://hitran.org/cia/",
    refresh: bool = False,
) -> Path:
    """Download a named CIA file into the requested local cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.exists() and not refresh:
        return target
    url = find_cia_download_url(index_url, filename)
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
        index_url=config.cia_index_url,
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
    index_url: str = "https://hitran.org/cia/",
    refresh: bool = False,
) -> CiaDataset:
    """Download and parse a CIA dataset identified by filename and pair."""
    path = download_cia_file_by_name(
        cache_dir=cache_dir,
        filename=filename,
        index_url=index_url,
        refresh=refresh,
    )
    return parse_cia_file(path, pair)


def plot_cia_cross_section(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    wavenumber_grid_cm1: np.ndarray,
    figure_path: Path,
) -> np.ndarray:
    """Plot a CIA binary absorption coefficient spectrum in cm^5 molecule^-2."""
    import matplotlib.pyplot as plt

    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    binary_xsec = dataset.interpolate_to_grid(temperature_k, wavenumber_grid_cm1)
    positive = binary_xsec[binary_xsec > 0.0]
    if positive.size == 0:
        raise ValueError("No positive CIA coefficients were found on the requested grid.")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavenumber_grid_cm1, binary_xsec, color="tab:blue", linewidth=1.25)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Binary absorption coefficient [cm$^5$ / molecule$^2$]")
    ax.set_title(f"{dataset.pair} CIA at {temperature_k:.1f} K")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return binary_xsec


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


def plot_cia_attenuation_coefficient(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    pressure_pa: float,
    wavenumber_grid_cm1: np.ndarray,
    figure_path: Path,
) -> np.ndarray:
    """Plot a CIA attenuation coefficient spectrum in 1/m."""
    import matplotlib.pyplot as plt

    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    attenuation_m1 = compute_cia_attenuation_m1(
        dataset,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        wavenumber_grid_cm1=wavenumber_grid_cm1,
    )
    positive = attenuation_m1[attenuation_m1 > 0.0]
    if positive.size == 0:
        raise ValueError("No positive CIA attenuation coefficients were found on the requested grid.")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavenumber_grid_cm1, attenuation_m1, color="tab:orange", linewidth=1.25)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Attenuation coefficient [1/m]")
    ax.set_title(f"{dataset.pair} CIA attenuation at {temperature_k:.1f} K and {pressure_pa / 1.0e5:.3f} bar")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return attenuation_m1


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


def plot_cia_transmission(
    dataset: CiaDataset,
    *,
    temperature_k: float,
    pressure_pa: float,
    path_length_m: float,
    wavenumber_grid_cm1: np.ndarray,
    figure_path: Path,
) -> np.ndarray:
    """Plot CIA transmission over a fixed path length."""
    import matplotlib.pyplot as plt

    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    transmission = compute_cia_transmission(
        dataset,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        path_length_m=path_length_m,
        wavenumber_grid_cm1=wavenumber_grid_cm1,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wavenumber_grid_cm1, transmission, color="tab:green", linewidth=1.25)
    ax.plot(
        wavenumber_grid_cm1,
        compute_normalized_blackbody_curve(
            wavenumber_cm1=wavenumber_grid_cm1,
            temperature_k=temperature_k,
        ),
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Blackbody",
    )
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Transmission")
    ax.set_ylim(0.0, 1.01)
    ax.set_title(
        f"{dataset.pair} CIA transmission at {temperature_k:.1f} K, "
        f"{pressure_pa / 1.0e5:.3f} bar, L={path_length_m / 1000.0:.3f} km"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return transmission
