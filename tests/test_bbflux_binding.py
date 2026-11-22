#!/usr/bin/env python3
"""Every default that a bbflux_wavenumber overload advertises can be used.

The (wn1: float, wn2: float, temp: Tensor) overload advertised
``temp: torch.Tensor = 1``. An int cannot become a Tensor, so the advertised
call ``bbflux_wavenumber(1.0, 2.0)`` raised TypeError. temp has no physical
default (the C++ declaration in src/radiation/bbflux.hpp and the stub in
python/pyharp.pyi take it as required), so the binding now requires it.
"""

import re

import pytest
import torch

import pyharp

BAND = re.compile(r"bbflux_wavenumber\(wn1: [^,]*float[^,]*, wn2: [^,]*float[^,]*, temp: ([^)]*)\)", re.I)


def band_overload_temp():
    m = BAND.search(pyharp.bbflux_wavenumber.__doc__)
    assert m, pyharp.bbflux_wavenumber.__doc__
    return m.group(1)


def test_band_overload_default_is_usable():
    if "=" in band_overload_temp():
        # an advertised default must work
        pyharp.bbflux_wavenumber(1.0, 2.0)
    else:
        with pytest.raises(TypeError):
            pyharp.bbflux_wavenumber(1.0, 2.0)


def test_band_overload_matches_the_stub():
    assert "=" not in band_overload_temp(), "temp is required (bbflux.hpp, pyharp.pyi)"


def test_band_overload_with_temp():
    temp = torch.tensor([250.0, 300.0], dtype=torch.float64)
    a = pyharp.bbflux_wavenumber(1.0, 2.0, temp)
    b = pyharp.bbflux_wavenumber(wn1=1.0, wn2=2.0, temp=temp)
    assert torch.equal(a, b)
    assert a.shape == temp.shape and bool((a > 0).all())
