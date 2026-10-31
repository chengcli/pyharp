"""Toon longwave reads the surface temperature from bc "btemp", as DISORT does; without it the surface is at the
temperature of the bottom level, as before."""

import math

import pytest
import torch

import pyharp


def toon(flags, temf, **bc):
    op = pyharp.ToonMcKay89Options()
    op.wave_lower([300.0])
    op.wave_upper([600.0])
    op.flags(flags)
    nlyr = temf.shape[-1] - 1
    prop = torch.zeros(1, 1, nlyr, 3, dtype=torch.float64)
    prop[..., 0] = 1.0
    return pyharp.ToonMcKay89(op)(prop, temf=temf, albedo=torch.zeros(1, 1, dtype=torch.float64), **bc)[0, 0]


def band_pib(temp):
    return math.pi * pyharp.bbflux_wavenumber(300.0, 600.0, torch.tensor([temp], dtype=torch.float64)).flatten()[0].item()


def test_hard_surface_emits_at_btemp():
    temf = torch.full((1, 5), 250.0, dtype=torch.float64)
    flux = toon("planck,hard_surface", temf, btemp=torch.tensor([300.0], dtype=torch.float64))
    # black surface: the upward flux leaving it is pi B(btemp); level 0 is the bottom
    assert abs(flux[0, 0].item() / band_pib(300.0) - 1.0) < 1e-12
    # the atmosphere's own emission is unchanged: the downward flux at the surface does not depend on btemp
    ref = toon("planck,hard_surface", temf)
    assert torch.equal(flux[:, 1], ref[:, 1])


@pytest.mark.parametrize("flags", ["planck", "planck,hard_surface"])
def test_btemp_equal_to_the_bottom_level_changes_nothing(flags):
    temf = torch.linspace(300.0, 200.0, 5, dtype=torch.float64).view(1, 5)
    ref = toon(flags, temf)
    same = toon(flags, temf, btemp=torch.tensor([300.0], dtype=torch.float64))
    assert torch.equal(ref, same)


def test_gas_giant_surface_source_uses_btemp():
    temf = torch.full((1, 5), 250.0, dtype=torch.float64)
    hot = toon("planck", temf, btemp=torch.tensor([300.0], dtype=torch.float64))
    ref = toon("planck", temf)
    assert hot[0, 0].item() > ref[0, 0].item() * 1.5


def test_btemp_shape_is_checked():
    temf = torch.full((1, 5), 250.0, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        toon("planck,hard_surface", temf, btemp=torch.tensor([300.0, 310.0], dtype=torch.float64))
