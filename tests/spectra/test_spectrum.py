from pathlib import Path

import numpy as np
import xarray as xr

from pyharp.spectra.hitran_cia_utils import CiaBlock, CiaDataset
from pyharp.spectra.spectrum import (
    compute_absorption_spectrum_from_sources,
    plot_absorption_spectrum,
    plot_attenuation_spectrum,
    spectrum_to_dataset,
    write_spectrum_dataset,
)


class FakeLineProvider:
    def cross_section_cm2_molecule(self, wavenumber_grid_cm1, temperature_k, pressure_pa):
        del temperature_k, pressure_pa
        return np.full_like(wavenumber_grid_cm1, 2.0e-20, dtype=np.float64)


def test_compute_absorption_spectrum_from_sources_includes_cia(tmp_path: Path) -> None:
    cia = CiaDataset(
        pair="CO2-CO2",
        source_path=tmp_path / "mock.cia",
        blocks=(
            CiaBlock(
                temperature_k=300.0,
                wavenumber_cm1=np.array([20.0, 21.0, 22.0]),
                binary_cross_section_cm5_molecule2=np.array([1.0e-46, 2.0e-46, 3.0e-46]),
            ),
        ),
    )
    spectrum = compute_absorption_spectrum_from_sources(
        wavenumber_grid_cm1=np.array([20.0, 21.0, 22.0]),
        temperature_k=300.0,
        pressure_pa=1.0e5,
        line_provider=FakeLineProvider(),
        cia_dataset=cia,
    )

    assert np.all(spectrum.sigma_line_cm2_molecule > 0.0)
    assert np.all(spectrum.sigma_cia_cm2_molecule > 0.0)
    assert np.allclose(
        spectrum.sigma_total_cm2_molecule,
        spectrum.sigma_line_cm2_molecule + spectrum.sigma_cia_cm2_molecule,
    )
    assert np.allclose(
        spectrum.kappa_total_cm1,
        spectrum.kappa_line_cm1 + spectrum.kappa_cia_cm1,
    )
    assert np.allclose(spectrum.attenuation_total_m1, spectrum.kappa_total_cm1 * 100.0)
    assert spectrum.species_name == "CO2"


def test_spectrum_outputs_can_be_written(tmp_path: Path) -> None:
    spectrum = compute_absorption_spectrum_from_sources(
        wavenumber_grid_cm1=np.array([20.0, 21.0, 22.0]),
        temperature_k=300.0,
        pressure_pa=1.0e5,
        line_provider=FakeLineProvider(),
        cia_dataset=CiaDataset(
            pair="CO2-CO2",
            source_path=tmp_path / "mock.cia",
            blocks=(
                CiaBlock(
                    temperature_k=300.0,
                    wavenumber_cm1=np.array([20.0, 21.0, 22.0]),
                    binary_cross_section_cm5_molecule2=np.array([1.0e-46, 1.0e-46, 1.0e-46]),
                ),
            ),
        ),
    )
    dataset = spectrum_to_dataset(spectrum)
    assert "sigma_total_cm2_molecule" in dataset.data_vars
    assert "kappa_total_cm1" in dataset.data_vars
    assert "attenuation_total_m1" in dataset.data_vars
    assert dataset.attrs["species_name"] == "CO2"

    output_path = tmp_path / "spectrum.nc"
    figure_path = tmp_path / "spectrum.png"
    write_spectrum_dataset(spectrum, output_path)
    plot_absorption_spectrum(spectrum, figure_path)
    attenuation_figure_path = tmp_path / "attenuation.png"
    plot_attenuation_spectrum(spectrum, attenuation_figure_path)

    loaded = xr.load_dataset(output_path)
    try:
        assert np.allclose(loaded["sigma_total_cm2_molecule"].values, spectrum.sigma_total_cm2_molecule)
        assert np.allclose(loaded["kappa_total_cm1"].values, spectrum.kappa_total_cm1)
        assert np.allclose(loaded["attenuation_total_m1"].values, spectrum.attenuation_total_m1)
    finally:
        loaded.close()
    assert figure_path.exists()
    assert attenuation_figure_path.exists()


def test_compute_absorption_spectrum_from_cross_section_continuum() -> None:
    spectrum = compute_absorption_spectrum_from_sources(
        species_name="H2O",
        wavenumber_grid_cm1=np.array([20.0, 21.0, 22.0]),
        temperature_k=300.0,
        pressure_pa=1.0e5,
        line_provider=FakeLineProvider(),
        cia_cross_section_cm2_molecule=np.array([1.0e-24, 2.0e-24, 3.0e-24]),
    )

    assert spectrum.species_name == "H2O"
    assert np.allclose(spectrum.kappa_cia_cm1, spectrum.sigma_cia_cm2_molecule * spectrum.number_density_cm3)
