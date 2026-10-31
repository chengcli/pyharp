"""Toon longwave: an isothermal column with an isothermal top boundary is in radiative equilibrium, so every
upward and downward flux equals pi B, for any single-scattering albedo, surface albedo and surface type."""

import math

import pytest
import torch

import pyharp


def isothermal_fluxes(flags, albedo, ssa, temp=250.0, nlyr=4, dtau=1.0):
    op = pyharp.ToonMcKay89Options()
    op.wave_lower([300.0])
    op.wave_upper([600.0])
    op.flags(flags)
    op.top_emission_flag(1)
    prop = torch.zeros(1, 1, nlyr, 3, dtype=torch.float64)
    prop[..., 0] = dtau
    prop[..., 1] = ssa
    return pyharp.ToonMcKay89(op)(prop, temf=torch.full((1, nlyr + 1), temp, dtype=torch.float64),
                                  albedo=torch.full((1, 1), albedo, dtype=torch.float64))[0, 0]


@pytest.mark.parametrize("flags", ["planck", "planck,hard_surface"])
@pytest.mark.parametrize("albedo", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("ssa", [0.0, 0.5, 0.9])
def test_isothermal_column_is_in_equilibrium(flags, albedo, ssa):
    pib = math.pi * pyharp.bbflux_wavenumber(300.0, 600.0, torch.tensor([250.0], dtype=torch.float64)).flatten()[0]
    flux = isothermal_fluxes(flags, albedo, ssa)
    err = (flux[:, :2] / pib - 1.0).abs().max().item()
    assert err < 1e-12, f"{flags}, albedo {albedo}, ssa {ssa}: max |F / (pi B) - 1| = {err:.3e}"
