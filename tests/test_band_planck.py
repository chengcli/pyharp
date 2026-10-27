#!/usr/bin/env python3
"""Thermal band fluxes against an independent Planck integral.

An isothermal column over a black surface at the same temperature emits
pi * int B_nu dnu at the top, for any optical thickness and either solver. The
reference is that integral evaluated by mpmath quadrature at 30 digits with the
exact SI constants (CODATA 2018).

DISORT takes its Planck function from pydisort (c_planck_func2 of cdisort, with
C2 = 1.438786 cm K and sigma = 5.67032e-8 W m^-2 K^-4 hard-coded, a series cut
after 2 to 6 terms and a Simpson rule converged to 1e-6 for narrow bands),
which pyharp does not control: on a 10 cm^-1 band at 250 K it is 3e-5 from
the integral. The DISORT bands are therefore compared with pi times a
line-for-line port of c_planck_func2, which isolates what pyharp does with the
solver's output; their distance to the integral is printed, not asserted.
"""

import math

import mpmath as mp
import pytest
import torch

import pyharp

torch.set_default_dtype(torch.float64)
mp.mp.dps = 30

H = mp.mpf("6.62607015e-34")  # J s
C = mp.mpf(299792458)  # m / s
K = mp.mpf("1.380649e-23")  # J / K
C2 = H * C / K * 100  # cm K
SIGMA = 2 * mp.pi**5 * K**4 / (15 * H**3 * C**2)  # W m^-2 K^-4

RTOL = 1e-6
TEMP = 250.0
BANDS = [(500.0, 510.0), (400.0, 600.0), (100.0, 1100.0)]  # 10, 200, 1000 cm^-1


def band_planck_flux(nu1, nu2, temp, c2=C2, sigma=SIGMA):
    """pi * int_nu1^nu2 B_nu dnu [W m^-2], nu in cm^-1, by quadrature of
    sigma T^4 15 / pi^4 int x^3 / (e^x - 1) dx over x = c2 nu / T"""
    temp = mp.mpf(temp)
    x1, x2 = c2 * mp.mpf(nu1) / temp, c2 * mp.mpf(nu2) / temp
    frac = 15 / mp.pi**4 * mp.quad(lambda x: x**3 / mp.expm1(x), [x1, x2])
    return float(sigma * temp**4 * frac)


def cdisort_planck_band(wnumlo, wnumhi, t):
    """c_planck_func2 of cdisort 2.1.3 as bundled with pydisort (planck.h),
    ported line for line [W m^-2 sr^-1]"""
    c2, sigma, vcut = 1.438786, 5.67032e-8, 1.5
    a1, a2, a3, a4, a5, a6 = 1 / 3, -1 / 8, 1 / 60, -1 / 5040, 1 / 272160, -1 / 13305600
    vcp = [10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0]
    sigdpi, conc, c1 = sigma / math.pi, 15.0 / math.pi**4, 1.1911e-8
    if wnumhi == wnumlo:
        arg = math.exp(-c2 * wnumhi / t)
        return c1 * wnumhi**3 * arg / (1.0 - arg)

    def plkf(x):
        return x**3 / (math.exp(x) - 1.0)

    v = [c2 * wnumlo / t, c2 * wnumhi / t]
    if v[0] > 2.220446049250313e-16 and v[1] < math.log(1.7976931348623157e308) \
            and (wnumhi - wnumlo) / wnumhi < 1e-2:
        hh, oldval, val0 = v[1] - v[0], 0.0, plkf(v[0]) + plkf(v[1])
        for n in range(1, 11):
            dl = hh / (2 * n)
            val = val0
            for k in range(1, 2 * n):
                val += 2 * (1 + k % 2) * plkf(v[0] + k * dl)
            val *= dl * a1
            if abs((val - oldval) / val) <= 1e-6:
                break
            oldval = val
        return sigdpi * (t * t) ** 2 * conc * val
    smallv, p, d = 0, [0.0, 0.0], [0.0, 0.0]
    for i in range(2):
        if v[i] < vcut:
            smallv += 1
            vsq = v[i] * v[i]
            p[i] = conc * vsq * v[i] * (a1 + v[i] * (a2 + v[i] * (a3 + vsq * (a4 + vsq * (a5 + vsq * a6)))))
        else:
            mmax = 1
            while v[i] < vcp[mmax - 1]:
                mmax += 1
            ex, exm = math.exp(-v[i]), 1.0
            for m in range(1, mmax + 1):
                mv = m * v[i]
                exm *= ex
                d[i] += exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / (m * m) ** 2
            d[i] *= conc
    ans = p[1] - p[0] if smallv == 2 else (1.0 - p[0] - d[1] if smallv == 1 else d[0] - d[1])
    return ans * sigdpi * (t * t) ** 2


def reference(solver, nu1, nu2, temp):
    if solver == "disort":
        return math.pi * cdisort_planck_band(nu1, nu2, temp)
    return band_planck_flux(nu1, nu2, temp)


def make_band(tmp_path, solver, nu1, nu2, nwave, nlyr=4):
    """A grey, purely absorbing band (fourcolumn table, constant extinction)
    on a wavenumber grid of nwave points between the band limits."""
    (tmp_path / "grey.txt").write_text("1. 1. 0. 0.\n100000. 1. 0. 0.\n")
    flags = "planck,hard_surface" if solver == "toon" else "lamber,quiet,onlyfl,planck"
    (tmp_path / "band.yaml").write_text(f"""
species:
  - name: dust
    composition: {{C: 1}}
geometry:
  cells: {{nx1: {nlyr}}}
opacities:
  grey: {{type: fourcolumn, data: [{tmp_path / "grey.txt"}], species: [dust], nmom: 0}}
bands:
  - name: B
    range: [{nu1!r}, {nu2!r}]
    opacities: [grey]
    solver: {solver}
    nwave: {nwave}
    nstr: 4
    flags: {flags}
""")
    options = pyharp.RadiationOptions.from_yaml(str(tmp_path / "band.yaml"))
    band = options.bands()[0]
    if nwave == 1:
        band.wavenumber([nu1])
    else:
        band.wavenumber(torch.linspace(nu1, nu2, nwave).tolist())
    if solver == "disort":
        pyharp.disort_config(band.disort(), 4, nlyr, 1, nwave)
    return pyharp.RadiationBand(band), nlyr


def toa_upward_flux(band, nlyr, nwave, temp):
    conc = torch.full((1, nlyr, 1), 1.0 / 12.011e-3)  # optical depth 1 per layer
    dz = torch.ones(nlyr)
    bc = {
        "B/albedo": torch.zeros(nwave, 1),
        "B/fbeam": torch.zeros(nwave, 1),
        "B/umu0": torch.ones(1),
        "btemp": torch.full((1,), temp),
        "ttemp": torch.zeros(1),
    }
    atm = {"tempf": torch.full((1, nlyr + 1), temp)}
    return band.forward(conc, dz, bc, atm)[0, -1, 0].item()


@pytest.mark.parametrize("solver", ["toon", "disort"])
@pytest.mark.parametrize("nu1,nu2", BANDS, ids=["10cm-1", "200cm-1", "1000cm-1"])
def test_wavenumber_grid_band_flux_is_the_planck_integral(tmp_path, solver, nu1, nu2):
    """P1: on a wavenumber grid every point carries the band-integrated Planck
    source, so the band flux is their average, not their integral over
    wavenumber (which is the band width times too large)."""
    nwave = 17
    band, nlyr = make_band(tmp_path, solver, nu1, nu2, nwave)
    flux = toa_upward_flux(band, nlyr, nwave, TEMP)
    ref = reference(solver, nu1, nu2, TEMP)
    rel = flux / ref - 1.0
    exact = band_planck_flux(nu1, nu2, TEMP)
    print(f"{solver} {nu1}-{nu2} cm-1: flux {flux:.9e}, reference {ref:.9e}, rel {rel:+.3e}; "
          f"rel to the integral {flux / exact - 1.0:+.3e}")
    assert abs(rel) < RTOL
