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


def make_band(tmp_path, solver, nu1, nu2, nwave, nlyr=4, thermal=True, descending=False):
    """A grey, purely absorbing band (fourcolumn table, constant extinction)
    on a wavenumber grid of nwave points between the band limits, ascending
    or descending; without thermal, the solver has no Planck source (a
    beam-only band)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "grey.txt").write_text("1. 1. 0. 0.\n100000. 1. 0. 0.\n")
    if thermal:
        flags = "planck,hard_surface" if solver == "toon" else "lamber,quiet,onlyfl,planck"
    else:
        flags = '""' if solver == "toon" else "lamber,quiet,onlyfl"
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
        grid = torch.linspace(nu1, nu2, nwave)
        band.wavenumber((grid.flip(0) if descending else grid).tolist())
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


def planck_radiance(nu, temp):
    """B_nu [W m^-2 sr^-1 (cm^-1)^-1], nu in cm^-1, exact SI constants"""
    nu, temp = mp.mpf(nu), mp.mpf(temp)
    return float(2 * H * C**2 * 1e8 * nu**3 / mp.expm1(C2 * nu / temp))


MONO = [10.0, 200.0, 1000.0]  # cm^-1


@pytest.mark.parametrize("nu", MONO, ids=["10cm-1", "200cm-1", "1000cm-1"])
def test_monochromatic_planck_radiance(nu):
    """P2: with equal edges bbflux_wavenumber returns the radiance per
    wavenumber at that wavenumber, and so does the point overload."""
    temp = torch.tensor([TEMP])
    ref = planck_radiance(nu, TEMP)
    got = {
        "scalar edges": pyharp.bbflux_wavenumber(nu, nu, temp).item(),
        "tensor edges": pyharp.bbflux_wavenumber(torch.tensor([nu]), torch.tensor([nu]), temp).item(),
        "point": pyharp.bbflux_wavenumber(torch.tensor([nu]), TEMP).item(),
    }
    for name, value in got.items():
        print(f"{name} {nu} cm-1: {value:.9e}, reference {ref:.9e}, rel {value / ref - 1.0:+.3e}")
    for value in got.values():
        assert abs(value / ref - 1.0) < RTOL

    # equal and unequal edges in one call: each element gets its own branch
    wn1, wn2 = torch.tensor([nu, nu]), torch.tensor([nu, nu + 10.0])
    mixed = pyharp.bbflux_wavenumber(wn1, wn2, temp)
    band = band_planck_flux(nu, nu + 10.0, TEMP) / math.pi
    assert abs(mixed[0].item() / ref - 1.0) < RTOL
    assert abs(mixed[1].item() / band - 1.0) < RTOL


@pytest.mark.parametrize("solver", ["toon", "disort"])
@pytest.mark.parametrize("nu", MONO, ids=["10cm-1", "200cm-1", "1000cm-1"])
def test_single_wavenumber_band_flux_is_pi_b_nu(tmp_path, solver, nu):
    """P2: a band whose range is one wavenumber gives the flux per wavenumber
    pi B_nu at the top of the isothermal column."""
    band, nlyr = make_band(tmp_path, solver, nu, nu, 1)
    flux = toa_upward_flux(band, nlyr, 1, TEMP)
    exact = math.pi * planck_radiance(nu, TEMP)
    ref = math.pi * cdisort_planck_band(nu, nu, TEMP) if solver == "disort" else exact
    rel = flux / ref - 1.0
    print(f"{solver} {nu} cm-1: flux {flux:.9e}, reference {ref:.9e}, rel {rel:+.3e}; "
          f"rel to pi B_nu {flux / exact - 1.0:+.3e}")
    assert abs(rel) < RTOL


# P11: the temperature that puts the edge 300 cm^-1 at v = c2 nu / T = 1.5
# exactly in double precision, with c2 as the code writes it
CODE_C2 = 1.438776877503933
T_SWITCH = CODE_C2 * 300.0 / 1.5
SWITCH_BANDS = [(300.0, 600.0), (150.0, 300.0)]  # lower and upper edge at v = 1.5
AROUND = [0.0, -1e-12, 1e-12, -1e-6, 1e-6, -1e-3, 1e-3]


def test_switch_temperature_puts_the_edge_at_v_one_and_a_half():
    assert CODE_C2 * 300.0 / T_SWITCH == 1.5


@pytest.mark.parametrize("nu1,nu2", SWITCH_BANDS, ids=["lower-edge", "upper-edge"])
def test_band_planck_at_and_around_the_series_switch(nu1, nu2):
    """P11: a band edge with v = 1.5 exactly is in one of the two series, and
    the band radiance is continuous across it."""
    temps = [T_SWITCH * (1.0 + e) for e in AROUND]
    tensor = pyharp.bbflux_wavenumber(torch.tensor([nu1]), torch.tensor([nu2]), torch.tensor(temps))
    for k, (e, t) in enumerate(zip(AROUND, temps)):
        ref = band_planck_flux(nu1, nu2, t) / math.pi
        scalar = pyharp.bbflux_wavenumber(nu1, nu2, torch.tensor([t])).item()
        print(f"{nu1}-{nu2} cm-1, T = T_switch (1 {e:+.0e}): scalar rel {scalar / ref - 1.0:+.3e}, "
              f"tensor rel {tensor[k].item() / ref - 1.0:+.3e}")
        assert abs(scalar / ref - 1.0) < RTOL
        assert abs(tensor[k].item() / ref - 1.0) < RTOL


@pytest.mark.parametrize("solver", ["toon", "disort"])
@pytest.mark.parametrize("nu1,nu2", SWITCH_BANDS, ids=["lower-edge", "upper-edge"])
def test_band_flux_with_an_edge_at_the_series_switch(tmp_path, solver, nu1, nu2):
    """P11: band flux of the isothermal column at T_switch."""
    nwave = 17
    band, nlyr = make_band(tmp_path, solver, nu1, nu2, nwave)
    flux = toa_upward_flux(band, nlyr, nwave, T_SWITCH)
    ref = reference(solver, nu1, nu2, T_SWITCH)
    rel = flux / ref - 1.0
    exact = band_planck_flux(nu1, nu2, T_SWITCH)
    print(f"{solver} {nu1}-{nu2} cm-1 at T_switch: flux {flux:.9e}, reference {ref:.9e}, rel {rel:+.3e}; "
          f"rel to the integral {flux / exact - 1.0:+.3e}")
    assert abs(rel) < RTOL


UMU0 = 0.5


def toa_downward_beam_flux(band, nlyr, nwave, fbeam):
    """Downward flux at the top of a beam-only band: umu0 times the beam
    irradiance, whatever the column below."""
    conc = torch.full((1, nlyr, 1), 0.1 / 12.011e-3)  # optical depth 0.1 per layer
    dz = torch.ones(nlyr)
    bc = {
        "B/albedo": torch.zeros(nwave, 1),
        "B/fbeam": torch.full((nwave, 1), fbeam),
        "B/umu0": torch.full((1,), UMU0),
    }
    # level temperatures only because RadiationBand calls DISORT with them;
    # without the planck flag the solvers do not use them
    atm = {"tempf": torch.full((1, nlyr + 1), 250.0)}
    return band.forward(conc, dz, bc, atm)[0, -1, 1].item()


@pytest.mark.parametrize("solver", ["toon", "disort"])
@pytest.mark.parametrize("nu1,nu2", BANDS[:2], ids=["10cm-1", "200cm-1"])
def test_beam_irradiance_is_per_point_band_irradiance(tmp_path, solver, nu1, nu2):
    """fbeam at each spectral point is the band's beam irradiance [W m^-2],
    in the units of the band-integrated Planck source the solver adds to it
    (DISORT: FBEAM in the units of PLKAVG), not a density per wavenumber. A
    beam-only band on an unweighted wavenumber grid then delivers umu0 * fbeam
    at the top, not the band width times it."""
    nwave, fbeam = 17, 1.0
    band, nlyr = make_band(tmp_path, solver, nu1, nu2, nwave, thermal=False)
    flux = toa_downward_beam_flux(band, nlyr, nwave, fbeam)
    print(f"{solver} {nu1}-{nu2} cm-1, fbeam {fbeam} at every point: TOA downward {flux:.12e}, "
          f"umu0 * fbeam {UMU0 * fbeam}")
    assert abs(flux / (UMU0 * fbeam) - 1.0) < 1e-12


@pytest.mark.parametrize("solver", ["toon", "disort"])
@pytest.mark.parametrize("thermal", [True, False], ids=["thermal", "beam"])
def test_descending_wavenumber_grid_gives_the_same_band_flux(tmp_path, solver, thermal):
    """The same band on an ascending and a descending wavenumber grid gives the
    same positive band flux."""
    nu1, nu2, nwave = BANDS[1][0], BANDS[1][1], 17
    flux = {}
    for order in ("ascending", "descending"):
        band, nlyr = make_band(tmp_path / order, solver, nu1, nu2, nwave, thermal=thermal,
                               descending=order == "descending")
        if thermal:
            flux[order] = toa_upward_flux(band, nlyr, nwave, TEMP)
        else:
            flux[order] = toa_downward_beam_flux(band, nlyr, nwave, 1.0)
    ref = reference(solver, nu1, nu2, TEMP) if thermal else UMU0
    print(f"{solver} {'thermal' if thermal else 'beam'} {nu1}-{nu2} cm-1: ascending {flux['ascending']:.12e}, "
          f"descending {flux['descending']:.12e}, reference {ref:.12e}")
    assert flux["descending"] > 0.0
    assert abs(flux["descending"] / flux["ascending"] - 1.0) < 1e-12
    assert abs(flux["ascending"] / ref - 1.0) < RTOL
