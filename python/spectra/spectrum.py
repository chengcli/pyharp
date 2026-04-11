"""Single-state absorption spectrum utilities and plotting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spectra_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .config import SpectroscopyConfig, SpectralBandConfig
from .dataset_io import write_dataset_via_tmp
from .hitran_cia import CiaDataset, download_cia_file, parse_cia_file
from .hitran_lines import HapiLineProvider, LineDatabase, build_line_provider, download_hitran_lines
from .mt_ckd_h2o import compute_mt_ckd_h2o_continuum_cross_section

K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True)
class AbsorptionSpectrum:
    """Single-state molecular cross sections and absorption coefficients."""

    species_name: str
    wavenumber_cm1: np.ndarray
    sigma_line_cm2_molecule: np.ndarray
    sigma_cia_cm2_molecule: np.ndarray
    sigma_total_cm2_molecule: np.ndarray
    kappa_line_cm1: np.ndarray
    kappa_cia_cm1: np.ndarray
    kappa_total_cm1: np.ndarray
    attenuation_line_m1: np.ndarray
    attenuation_cia_m1: np.ndarray
    attenuation_total_m1: np.ndarray
    temperature_k: float
    pressure_pa: float
    number_density_cm3: float


def number_density_cm3_from_pressure_temperature(pressure_pa: float, temperature_k: float) -> float:
    """Convert thermodynamic state to number density in molecule/cm^3."""
    return float(pressure_pa / (K_BOLTZMANN * temperature_k) / 1.0e6)


def compute_absorption_spectrum_from_sources(
    *,
    species_name: str = "CO2",
    wavenumber_grid_cm1: np.ndarray,
    temperature_k: float,
    pressure_pa: float,
    line_provider: HapiLineProvider,
    cia_dataset: CiaDataset | None = None,
    cia_cross_section_cm2_molecule: np.ndarray | None = None,
) -> AbsorptionSpectrum:
    """Compute line, CIA, and total cross sections and absorption coefficients on one grid."""
    wavenumber_grid_cm1 = np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    sigma_line_cm2_molecule = line_provider.cross_section_cm2_molecule(
        wavenumber_grid_cm1=wavenumber_grid_cm1,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
    )
    number_density_cm3 = number_density_cm3_from_pressure_temperature(
        pressure_pa=pressure_pa,
        temperature_k=temperature_k,
    )
    kappa_line_cm1 = sigma_line_cm2_molecule * number_density_cm3
    if cia_cross_section_cm2_molecule is not None:
        sigma_cia_cm2_molecule = np.asarray(cia_cross_section_cm2_molecule, dtype=np.float64)
        if sigma_cia_cm2_molecule.shape != wavenumber_grid_cm1.shape:
            raise ValueError("cia_cross_section_cm2_molecule must match the wavenumber grid shape.")
        kappa_cia_cm1 = sigma_cia_cm2_molecule * number_density_cm3
    elif cia_dataset is not None:
        cia_binary_xsec_cm5_molecule2 = cia_dataset.interpolate_to_grid(
            temperature_k=temperature_k,
            wavenumber_grid_cm1=wavenumber_grid_cm1,
        )
        kappa_cia_cm1 = cia_binary_xsec_cm5_molecule2 * number_density_cm3**2
        sigma_cia_cm2_molecule = kappa_cia_cm1 / number_density_cm3
    else:
        sigma_cia_cm2_molecule = np.zeros_like(wavenumber_grid_cm1, dtype=np.float64)
        kappa_cia_cm1 = np.zeros_like(wavenumber_grid_cm1, dtype=np.float64)
    return AbsorptionSpectrum(
        species_name=str(species_name).upper(),
        wavenumber_cm1=wavenumber_grid_cm1,
        sigma_line_cm2_molecule=np.asarray(sigma_line_cm2_molecule, dtype=np.float64),
        sigma_cia_cm2_molecule=np.asarray(sigma_cia_cm2_molecule, dtype=np.float64),
        sigma_total_cm2_molecule=np.asarray(sigma_line_cm2_molecule + sigma_cia_cm2_molecule, dtype=np.float64),
        kappa_line_cm1=np.asarray(kappa_line_cm1, dtype=np.float64),
        kappa_cia_cm1=np.asarray(kappa_cia_cm1, dtype=np.float64),
        kappa_total_cm1=np.asarray(kappa_line_cm1 + kappa_cia_cm1, dtype=np.float64),
        attenuation_line_m1=np.asarray(kappa_line_cm1 * 100.0, dtype=np.float64),
        attenuation_cia_m1=np.asarray(kappa_cia_cm1 * 100.0, dtype=np.float64),
        attenuation_total_m1=np.asarray((kappa_line_cm1 + kappa_cia_cm1) * 100.0, dtype=np.float64),
        temperature_k=float(temperature_k),
        pressure_pa=float(pressure_pa),
        number_density_cm3=number_density_cm3,
    )


def _resolve_continuum_sources(
    *,
    config: SpectroscopyConfig,
    wavenumber_grid_cm1: np.ndarray,
    temperature_k: float,
    pressure_pa: float,
) -> tuple[CiaDataset | None, np.ndarray | None]:
    """Resolve optional continuum inputs for a spectrum calculation."""
    cia_dataset: CiaDataset | None = None
    cia_cross_section_cm2_molecule: np.ndarray | None = None
    if config.hitran_species.name == "H2O":
        cia_cross_section_cm2_molecule = compute_mt_ckd_h2o_continuum_cross_section(
            wavenumber_grid_cm1=wavenumber_grid_cm1,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            h2o_vmr=1.0,
        )
    elif config.hitran_species.cia_filename is not None:
        cia_path = download_cia_file(config)
        cia_dataset = parse_cia_file(cia_path, config.cia_pair)
    return cia_dataset, cia_cross_section_cm2_molecule


def compute_absorption_spectrum(
    config: SpectroscopyConfig,
    band: SpectralBandConfig,
    temperature_k: float,
    pressure_pa: float,
    *,
    line_db: LineDatabase | None = None,
) -> AbsorptionSpectrum:
    """Download/load spectroscopy inputs and compute one-state absorption."""
    config.ensure_directories()
    grid = band.grid()
    line_db = line_db or download_hitran_lines(config, band)
    line_provider = build_line_provider(config, line_db)
    cia_dataset, cia_cross_section_cm2_molecule = _resolve_continuum_sources(
        config=config,
        wavenumber_grid_cm1=grid,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
    )
    return compute_absorption_spectrum_from_sources(
        species_name=config.hitran_species.name,
        wavenumber_grid_cm1=grid,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        line_provider=line_provider,
        cia_dataset=cia_dataset,
        cia_cross_section_cm2_molecule=cia_cross_section_cm2_molecule,
    )


def spectrum_to_dataset(spectrum: AbsorptionSpectrum) -> xr.Dataset:
    """Convert a single-state absorption spectrum to an xarray dataset."""
    return xr.Dataset(
        coords={"wavenumber_cm1": ("wavenumber_cm1", spectrum.wavenumber_cm1)},
        data_vars={
            "sigma_line_cm2_molecule": ("wavenumber_cm1", spectrum.sigma_line_cm2_molecule),
            "sigma_cia_cm2_molecule": ("wavenumber_cm1", spectrum.sigma_cia_cm2_molecule),
            "sigma_total_cm2_molecule": ("wavenumber_cm1", spectrum.sigma_total_cm2_molecule),
            "kappa_line_cm1": ("wavenumber_cm1", spectrum.kappa_line_cm1),
            "kappa_cia_cm1": ("wavenumber_cm1", spectrum.kappa_cia_cm1),
            "kappa_total_cm1": ("wavenumber_cm1", spectrum.kappa_total_cm1),
            "attenuation_line_m1": ("wavenumber_cm1", spectrum.attenuation_line_m1),
            "attenuation_cia_m1": ("wavenumber_cm1", spectrum.attenuation_cia_m1),
            "attenuation_total_m1": ("wavenumber_cm1", spectrum.attenuation_total_m1),
        },
        attrs={
            "species_name": spectrum.species_name,
            "temperature_k": spectrum.temperature_k,
            "pressure_pa": spectrum.pressure_pa,
            "pressure_bar": spectrum.pressure_pa / 1.0e5,
            "number_density_cm3": spectrum.number_density_cm3,
        },
    )


def write_spectrum_dataset(spectrum: AbsorptionSpectrum, output_path: Path) -> None:
    """Write a single-state absorption spectrum dataset to NetCDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = spectrum_to_dataset(spectrum)
    try:
        write_dataset_via_tmp(dataset, output_path)
    finally:
        dataset.close()


def plot_absorption_spectrum(spectrum: AbsorptionSpectrum, figure_path: Path) -> None:
    """Plot line, CIA/continuum, and total absorption cross sections."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spectrum.wavenumber_cm1, spectrum.sigma_line_cm2_molecule, label="Line")
    ax.plot(spectrum.wavenumber_cm1, spectrum.sigma_cia_cm2_molecule, label="CIA / Continuum")
    ax.plot(spectrum.wavenumber_cm1, spectrum.sigma_total_cm2_molecule, label="Total", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Absorption cross section [cm$^2$ / molecule]")
    ax.set_title(
        f"{spectrum.species_name} absorption at T={spectrum.temperature_k:.1f} K, "
        f"p={spectrum.pressure_pa / 1.0e5:.3f} bar"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_attenuation_spectrum(spectrum: AbsorptionSpectrum, figure_path: Path) -> None:
    """Plot line, CIA/continuum, and total attenuation coefficients."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spectrum.wavenumber_cm1, spectrum.attenuation_line_m1, label="Line")
    ax.plot(spectrum.wavenumber_cm1, spectrum.attenuation_cia_m1, label="CIA / Continuum")
    ax.plot(spectrum.wavenumber_cm1, spectrum.attenuation_total_m1, label="Total", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Attenuation coefficient [1/m]")
    ax.set_title(
        f"{spectrum.species_name} attenuation at T={spectrum.temperature_k:.1f} K, "
        f"p={spectrum.pressure_pa / 1.0e5:.3f} bar"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
