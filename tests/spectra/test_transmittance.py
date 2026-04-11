from pathlib import Path

import numpy as np
import xarray as xr

from pyharp.spectra.blackbody import compute_normalized_blackbody_curve
from pyharp.spectra.spectrum import AbsorptionSpectrum
from pyharp.spectra.transmittance import (
    compute_transmittance_spectrum,
    plot_transmittance_spectrum,
    transmittance_to_dataset,
    write_transmittance_dataset,
)


def make_spectrum() -> AbsorptionSpectrum:
    return AbsorptionSpectrum(
        species_name="CO2",
        wavenumber_cm1=np.array([2000.0, 2001.0, 2002.0]),
        sigma_line_cm2_molecule=np.array([1.0e-20, 2.0e-20, 3.0e-20]),
        sigma_cia_cm2_molecule=np.array([1.0e-24, 1.0e-24, 1.0e-24]),
        sigma_total_cm2_molecule=np.array([1.0001e-20, 2.0001e-20, 3.0001e-20]),
        kappa_line_cm1=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        kappa_cia_cm1=np.array([1.0e-5, 1.0e-5, 1.0e-5]),
        kappa_total_cm1=np.array([1.01e-3, 2.01e-3, 3.01e-3]),
        attenuation_line_m1=np.array([0.1, 0.2, 0.3]),
        attenuation_cia_m1=np.array([0.001, 0.001, 0.001]),
        attenuation_total_m1=np.array([0.101, 0.201, 0.301]),
        temperature_k=300.0,
        pressure_pa=1.0e5,
        number_density_cm3=2.4e19,
    )


def test_compute_transmittance_spectrum() -> None:
    trans = compute_transmittance_spectrum(spectrum=make_spectrum(), path_length_m=1.0)
    assert np.allclose(trans.transmittance_line, np.exp(-np.array([0.1, 0.2, 0.3])))
    assert np.allclose(trans.transmittance_total, np.exp(-np.array([0.101, 0.201, 0.301])))


def test_transmittance_outputs_can_be_written(tmp_path: Path) -> None:
    trans = compute_transmittance_spectrum(spectrum=make_spectrum(), path_length_m=1.0)
    dataset = transmittance_to_dataset(trans)
    assert "transmittance_total" in dataset.data_vars

    output_path = tmp_path / "nested" / "trans.nc"
    figure_path = tmp_path / "trans.png"
    write_transmittance_dataset(trans, output_path)
    plot_transmittance_spectrum(trans, figure_path)

    loaded = xr.load_dataset(output_path)
    try:
        assert np.allclose(loaded["transmittance_total"].values, trans.transmittance_total)
    finally:
        loaded.close()
    assert figure_path.exists()


def test_normalized_blackbody_curve_is_bounded_and_peaks_at_one() -> None:
    curve = compute_normalized_blackbody_curve(
        wavenumber_cm1=np.array([100.0, 500.0, 1000.0, 1500.0]),
        temperature_k=300.0,
    )
    assert np.all(curve >= 0.0)
    assert np.all(curve <= 1.0)
    assert np.isclose(curve.max(), 1.0)
