from pathlib import Path

import torch

import pyharp
from pyharp.opacity import MieWaterLiquid


CONFIG = Path(__file__).with_name("water_liquid_mie_test.yaml")


def test_python_binding_and_yaml_factory():
    options = pyharp.RadiationOptions.from_yaml(str(CONFIG))
    band_options = options.bands()[0]
    band_options.wavenumber([20000.0]).weight([1.0])

    opacity = MieWaterLiquid(band_options.opacities()["water-liquid"])
    concentration = torch.tensor([[[0.01]]], dtype=torch.float64)
    atmosphere = {
        "wavenumber": torch.tensor([20000.0], dtype=torch.float64),
        "re": torch.tensor(10.0, dtype=torch.float64),
    }
    direct = opacity.forward(concentration, atmosphere)
    assert direct.shape == (1, 1, 1, 5)
    assert torch.isfinite(direct).all()
    assert direct[0, 0, 0, 0] > 0

    band = pyharp.RadiationBand(band_options)
    assert band.module("water-liquid") is not None
    boundary = {
        "shortwave/fbeam": torch.ones((1, 1), dtype=torch.float64),
        "shortwave/umu0": torch.tensor([0.5], dtype=torch.float64),
        "shortwave/albedo": torch.zeros((1, 1), dtype=torch.float64),
    }
    band.forward(
        concentration,
        torch.tensor([2.0], dtype=torch.float64),
        boundary,
        {"re": atmosphere["re"]},
    )
    prop = band.buffer("prop")
    assert torch.allclose(prop[..., 0], 2.0 * direct[..., 0])
    assert torch.allclose(prop[..., 1:], direct[..., 1:])
