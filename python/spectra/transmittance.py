"""Single-state transmittance utilities and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .blackbody import compute_normalized_blackbody_curve
from .config import SpectroscopyConfig, SpectralBandConfig
from .spectrum import AbsorptionSpectrum, compute_absorption_spectrum


@dataclass(frozen=True)
class TransmittanceSpectrum:
    """Beer-Lambert transmittance for a homogeneous molecular path."""

    species_name: str
    wavenumber_cm1: np.ndarray
    transmittance_line: np.ndarray
    transmittance_cia: np.ndarray
    transmittance_total: np.ndarray
    path_length_m: float
    temperature_k: float
    pressure_pa: float


def compute_transmittance_spectrum(
    *,
    spectrum: AbsorptionSpectrum,
    path_length_m: float,
) -> TransmittanceSpectrum:
    """Convert attenuation coefficients into transmittance over a fixed path."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be positive")
    return TransmittanceSpectrum(
        species_name=str(spectrum.species_name).upper(),
        wavenumber_cm1=np.asarray(spectrum.wavenumber_cm1, dtype=np.float64),
        transmittance_line=np.exp(-np.asarray(spectrum.attenuation_line_m1, dtype=np.float64) * path_length_m),
        transmittance_cia=np.exp(-np.asarray(spectrum.attenuation_cia_m1, dtype=np.float64) * path_length_m),
        transmittance_total=np.exp(-np.asarray(spectrum.attenuation_total_m1, dtype=np.float64) * path_length_m),
        path_length_m=float(path_length_m),
        temperature_k=float(spectrum.temperature_k),
        pressure_pa=float(spectrum.pressure_pa),
    )


def compute_transmittance_from_config(
    *,
    config: SpectroscopyConfig,
    band: SpectralBandConfig,
    temperature_k: float,
    pressure_pa: float,
    path_length_m: float,
) -> TransmittanceSpectrum:
    """Compute a transmittance spectrum from spectroscopy inputs."""
    spectrum = compute_absorption_spectrum(
        config=config,
        band=band,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
    )
    return compute_transmittance_spectrum(spectrum=spectrum, path_length_m=path_length_m)


def transmittance_to_dataset(transmittance: TransmittanceSpectrum) -> xr.Dataset:
    """Convert a transmittance spectrum to an xarray dataset."""
    return xr.Dataset(
        coords={"wavenumber_cm1": ("wavenumber_cm1", transmittance.wavenumber_cm1)},
        data_vars={
            "transmittance_line": ("wavenumber_cm1", transmittance.transmittance_line),
            "transmittance_cia": ("wavenumber_cm1", transmittance.transmittance_cia),
            "transmittance_total": ("wavenumber_cm1", transmittance.transmittance_total),
        },
        attrs={
            "species_name": transmittance.species_name,
            "path_length_m": transmittance.path_length_m,
            "temperature_k": transmittance.temperature_k,
            "pressure_pa": transmittance.pressure_pa,
            "pressure_bar": transmittance.pressure_pa / 1.0e5,
        },
    )


def write_transmittance_dataset(transmittance: TransmittanceSpectrum, output_path: Path) -> None:
    """Write a transmittance spectrum dataset to NetCDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = transmittance_to_dataset(transmittance)
    dataset.to_netcdf(output_path)
    dataset.close()


def plot_transmittance_spectrum(transmittance: TransmittanceSpectrum, figure_path: Path) -> None:
    """Plot line, CIA/continuum, and total transmittance."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(transmittance.wavenumber_cm1, transmittance.transmittance_line, label="Line")
    ax.plot(transmittance.wavenumber_cm1, transmittance.transmittance_cia, label="CIA / Continuum")
    ax.plot(transmittance.wavenumber_cm1, transmittance.transmittance_total, label="Total", linewidth=2.0)
    ax.plot(
        transmittance.wavenumber_cm1,
        compute_normalized_blackbody_curve(
            wavenumber_cm1=transmittance.wavenumber_cm1,
            temperature_k=transmittance.temperature_k,
        ),
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Blackbody",
    )
    ax.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax.set_ylabel("Transmittance")
    ax.set_ylim(0.0, 1.01)
    ax.set_title(
        f"{transmittance.species_name} transmittance at T={transmittance.temperature_k:.1f} K, "
        f"p={transmittance.pressure_pa / 1.0e5:.3f} bar, L={transmittance.path_length_m:.2f} m"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
