from pathlib import Path
import math

import torch

import pyharp
from pyharp.opacity import TemperatureSwitchWaterCloud


CONFIG = Path(__file__).with_name("water_cloud_temperature_test.yaml")


def test_temperature_switch_python_binding_and_yaml_factory():
    options = pyharp.RadiationOptions.from_yaml(str(CONFIG))
    band_options = options.bands()[0]
    band_options.wavenumber([20000.0]).weight([1.0])

    opacity = TemperatureSwitchWaterCloud(
        band_options.opacities()["water-cloud"]
    )
    concentration = torch.full((1, 2, 1), 0.01, dtype=torch.float64)
    atmosphere = {
        "wavenumber": torch.tensor([20000.0], dtype=torch.float64),
        "re": torch.tensor(
            3.0 * math.sqrt(3.0) * 50.0 / 8.0, dtype=torch.float64
        ),
        "temp": torch.tensor([[260.0, 280.0]], dtype=torch.float64),
    }
    direct = opacity.forward(concentration, atmosphere)
    assert direct.shape == (1, 1, 2, 4)
    assert torch.isfinite(direct).all()
    assert torch.all(direct[..., 0] > 0)

    band = pyharp.RadiationBand(band_options)
    assert band.module("water-cloud") is not None
    boundary = {
        "shortwave/fbeam": torch.ones((1, 1), dtype=torch.float64),
        "shortwave/umu0": torch.tensor([0.5], dtype=torch.float64),
        "shortwave/albedo": torch.zeros((1, 1), dtype=torch.float64),
    }
    band.forward(
        concentration,
        torch.full((2,), 2.0, dtype=torch.float64),
        boundary,
        {"re": atmosphere["re"], "temp": atmosphere["temp"]},
    )
    prop = band.buffer("prop")
    assert torch.allclose(prop[..., 0], 2.0 * direct[..., 0])
    assert torch.allclose(prop[..., 1:], direct[..., 1:])
