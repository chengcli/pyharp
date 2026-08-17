from pathlib import Path

import pytest
import torch

import pyharp
from pyharp.opacity import Rayleigh


CONFIG = Path(__file__).with_name("rayleigh_test.yaml")


def configured_options():
    options = pyharp.RadiationOptions.from_yaml(str(CONFIG))
    band = options.bands()[0]
    band.wavenumber([20000.0])
    band.weight([1.0])
    band.set_wave_lower([19999.5])
    band.set_wave_upper([20000.5])
    pyharp.disort_config(band.disort(), 4, 2, 1, 1)
    return options, band


def h2_cross_section_m2_per_mol(wavenumber_cm1: float) -> float:
    wavelength_angstrom = 1.0e8 / wavenumber_cm1
    sigma_cm2_per_molecule = (
        8.14e-13 / wavelength_angstrom**4
        + 1.28e-6 / wavelength_angstrom**6
        + 1.61 / wavelength_angstrom**8
    )
    return sigma_cm2_per_molecule * 6.02214179e23 * 1.0e-4


def test_rayleigh_python_binding_matches_reference_cross_section():
    _, band = configured_options()
    rayleigh = Rayleigh(band.opacities()["rayleigh"])
    concentration = torch.tensor([[[2.0], [3.0]]], dtype=torch.float64)
    result = rayleigh.forward(
        concentration,
        {"wavenumber": torch.tensor([20000.0], dtype=torch.float64)},
    )

    expected_sigma = h2_cross_section_m2_per_mol(20000.0)
    assert result.shape == (1, 1, 2, 6)
    assert result[0, 0, 0, 0].item() == pytest.approx(2.0 * expected_sigma)
    assert result[0, 0, 1, 0].item() == pytest.approx(3.0 * expected_sigma)
    assert torch.equal(result[..., 1], torch.ones_like(result[..., 1]))
    assert torch.equal(result[..., 2], torch.zeros_like(result[..., 2]))
    assert torch.equal(result[..., 3], 0.1 * torch.ones_like(result[..., 3]))
    assert torch.equal(result[..., 4:], torch.zeros_like(result[..., 4:]))


def test_radiation_band_regularizes_conservative_ssa_for_disort():
    _, band_options = configured_options()
    band = pyharp.RadiationBand(band_options)

    # alpha is below 1e-10 m^-1, which guards against a fixed-epsilon bias in
    # the opacity mixing step.
    concentration = torch.full((1, 2, 1), 1.0e-6, dtype=torch.float64)
    dz = torch.ones(2, dtype=torch.float64)
    boundary = {
        "shortwave/fbeam": torch.ones((1, 1), dtype=torch.float64),
        "shortwave/umu0": torch.tensor([0.5], dtype=torch.float64),
        "shortwave/albedo": torch.zeros((1, 1), dtype=torch.float64),
    }

    atmosphere = {
        "temp": torch.full((1, 2), 300.0, dtype=torch.float64),
    }
    spectrum = band.forward(concentration, dz, boundary, atmosphere)
    prop = band.buffer("prop")

    assert torch.all((prop[..., 0] > 0.0) & (prop[..., 0] < 1.0e-10))
    expected_ssa = torch.full_like(prop[..., 1], 1.0 - 1.0e-12)
    assert torch.equal(prop[..., 1], expected_ssa)
    assert torch.equal(prop[..., 2], torch.zeros_like(prop[..., 2]))
    assert torch.equal(prop[..., 3], 0.1 * torch.ones_like(prop[..., 3]))
    assert torch.isfinite(spectrum).all()
