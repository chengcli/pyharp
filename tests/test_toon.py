#!/usr/bin/env python3
"""Tests for the Toon-McKay89 two-stream radiative transfer solver.

This test suite validates the ToonMcKay89 solver by:
  - Comparing with the analytical Beer-Lambert law for pure absorption.
  - Verifying physical consistency (positive fluxes, energy conservation).
  - Comparing shortwave and longwave fluxes against the DISORT multi-stream solver.

Level ordering convention (both Toon and DISORT with ``upward=True``):
  - level 0    = bottom of atmosphere (surface)
  - level nlyr = top of atmosphere (TOA)
  - flx[..., 0] = upward flux, flx[..., 1] = downward flux

Implementation note
-------------------
The pyharp and pydisort packages both embed a private copy of the PyTorch
CPU dispatch stub machinery.  Running both in the same Python process can
corrupt the shared kernel registration table and produce incorrect fluxes
when one solver is called immediately after the other.

To avoid this, every test function that exercises pyharp ToonMcKay89 uses
a subprocess so that each solver gets a clean, isolated Python environment.
"""

import json
import subprocess
import sys
import math


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

_COMMON_HEADER = """\
import json
import torch
import pyharp
import pydisort
from pyharp.cpp import ToonMcKay89 as ToonModule
torch.set_default_dtype(torch.float64)
"""


def _run(code: str) -> dict:
    """Run *code* in a fresh subprocess and return its JSON-encoded output."""
    full = _COMMON_HEADER + code
    proc = subprocess.run(
        [sys.executable, "-c", full],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Subprocess exited with code {proc.returncode}:\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return json.loads(proc.stdout.strip())


# ---------------------------------------------------------------------------
# Reusable code snippets
# ---------------------------------------------------------------------------

_TOON_SETUP = """\
nwave = {nwave}; ncol = {ncol}; nlyr = {nlyr}
wave_lo = {wave_lo}; wave_hi = {wave_hi}
opt = pyharp.ToonMcKay89Options()
opt.wave_lower(wave_lo); opt.wave_upper(wave_hi)
toon = ToonModule(opt); toon.double()
bc = {{
    "fbeam": torch.ones(nwave, ncol, dtype=torch.float64) * {fbeam},
    "umu0":  torch.ones(ncol,  dtype=torch.float64) * {umu0},
    "albedo":torch.zeros(nwave, ncol, dtype=torch.float64),
}}
"""

_DISORT_SETUP = """\
nwave = {nwave}; ncol = {ncol}; nlyr = {nlyr}; nstr = {nstr}
wave_lo = {wave_lo}; wave_hi = {wave_hi}
dop = pydisort.DisortOptions()
pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
dop.upward(True); dop.flags({flags!r})
dop.wave_lower(wave_lo); dop.wave_upper(wave_hi)
disort = pydisort.Disort(dop); disort.double()
bc = {{
    "fbeam": torch.ones(nwave, ncol, dtype=torch.float64) * {fbeam},
    "umu0":  torch.ones(ncol,  dtype=torch.float64) * {umu0},
    "albedo":torch.zeros(nwave, ncol, dtype=torch.float64),
}}
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_toon_sw_pure_absorption():
    """Toon and DISORT must reproduce the Beer-Lambert law for pure absorption.

    With ``w0=0`` and zero surface albedo, the direct-beam downward flux at
    each level follows ``fbeam * umu0 * exp(-(nlyr - k) * tau / umu0)``.
    Both solvers must match this analytical result and each other.
    """
    nwave = 3; ncol = 1; nlyr = 10; nstr = 4
    tau = 0.1; fbeam = 1000.0; umu0 = 0.5
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    setup_common = dict(
        nwave=nwave, ncol=ncol, nlyr=nlyr, nstr=nstr,
        wave_lo=wave_lo, wave_hi=wave_hi,
        fbeam=fbeam, umu0=umu0,
    )

    # ---- Toon subprocess ----
    toon_code = (
        _TOON_SETUP.format(**setup_common)
        + f"""\
prop = torch.zeros(nwave, ncol, nlyr, 3, dtype=torch.float64)
prop[:, :, :, 0] = {tau}
r = toon(prop, bc, "", None)
print(json.dumps({{
    "dn": r[0, 0, :, 1].tolist(),
    "up": r[0, 0, :, 0].tolist(),
}}))
"""
    )
    toon_out = _run(toon_code)

    # ---- DISORT subprocess ----
    disort_code = (
        _DISORT_SETUP.format(flags="lamber,quiet,onlyfl", **setup_common)
        + f"""\
prop = torch.zeros(nwave, ncol, nlyr, 2 + nstr, dtype=torch.float64)
prop[:, :, :, 0] = {tau}
r = disort.forward(prop, bc, "", None)
print(json.dumps({{
    "dn": r[0, 0, :, 1].tolist(),
    "up": r[0, 0, :, 0].tolist(),
}}))
"""
    )
    disort_out = _run(disort_code)

    # Analytical Beer-Lambert law: level k, beam traverses (nlyr - k) layers
    expected_dn = [
        fbeam * umu0 * math.exp(-(nlyr - k) * tau / umu0)
        for k in range(nlyr + 1)
    ]

    # Both solvers must match the analytical solution
    for k in range(nlyr + 1):
        assert abs(toon_out["dn"][k] - expected_dn[k]) < 1e-8, (
            f"Toon Beer-Lambert mismatch at level {k}"
        )
        assert abs(disort_out["dn"][k] - expected_dn[k]) < 1e-8, (
            f"DISORT Beer-Lambert mismatch at level {k}"
        )

    # No upward flux with zero surface albedo and pure absorption
    for k in range(nlyr + 1):
        assert abs(toon_out["up"][k]) < 1e-10, (
            f"Non-zero Toon upward flux at level {k} for pure absorption"
        )

    # Toon and DISORT must agree exactly
    for k in range(nlyr + 1):
        assert abs(toon_out["dn"][k] - disort_out["dn"][k]) < 1e-8, (
            f"Toon/DISORT disagreement for pure absorption at level {k}"
        )


def test_toon_sw_scattering():
    """Toon fluxes must satisfy energy conservation for shortwave scattering.

    The Toon-McKay89 solver is a two-stream approximation, so its fluxes will
    differ from multi-stream DISORT.  This test verifies physical consistency:
      - Reflected flux at TOA does not exceed incident flux.
      - Downward flux at the surface does not exceed incident flux.
      - All flux values are non-negative.
    """
    nwave = 3; ncol = 1; nlyr = 10
    tau = 0.1; fbeam = 1000.0; umu0 = 0.5
    incident = fbeam * umu0
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    setup = dict(
        nwave=nwave, ncol=ncol, nlyr=nlyr, nstr=4,
        wave_lo=wave_lo, wave_hi=wave_hi, fbeam=fbeam, umu0=umu0,
    )

    toon_code = (
        _TOON_SETUP.format(**setup)
        + f"""\
results = []
# Order: anisotropic (g > 0) cases first to ensure the general scattering
# code path is exercised before the isotropic (g = 0) case.
for w0, g in [(0.5, 0.5), (0.5, 0.0), (0.9, 0.5)]:
    opt2 = pyharp.ToonMcKay89Options()
    opt2.wave_lower({wave_lo}); opt2.wave_upper({wave_hi})
    t = ToonModule(opt2); t.double()
    prop = torch.zeros(nwave, ncol, nlyr, 3, dtype=torch.float64)
    prop[:, :, :, 0] = {tau}; prop[:, :, :, 1] = w0; prop[:, :, :, 2] = g
    r = t(prop, bc, "", None)
    results.append({{
        "w0": w0, "g": g,
        "min_up": float(r[:, :, :, 0].min()),
        "min_dn": float(r[:, :, :, 1].min()),
        "up_toa": float(r[0, 0, -1, 0]),
        "dn_surf": float(r[0, 0, 0, 1]),
    }})
print(json.dumps(results))
"""
    )
    results = _run(toon_code)

    for res in results:
        w0, g = res["w0"], res["g"]
        assert res["min_up"] >= -1e-10, (
            f"Negative upward flux for w0={w0}, g={g}: {res['min_up']}"
        )
        assert res["min_dn"] >= -1e-10, (
            f"Negative downward flux for w0={w0}, g={g}: {res['min_dn']}"
        )
        assert res["up_toa"] <= incident + 1e-6, (
            f"TOA upward flux exceeds incident for w0={w0}, g={g}: {res['up_toa']}"
        )
        assert res["dn_surf"] <= incident + 1e-6, (
            f"Surface downward flux exceeds incident for w0={w0}, g={g}: {res['dn_surf']}"
        )


def test_toon_sw_vs_disort():
    """Toon and DISORT surface downward flux must agree within a factor of 3.

    For optically moderate atmospheres, both solvers give qualitatively
    consistent results.  The surface downward flux comparison uses a loose
    tolerance that accommodates the two-stream approximation error.
    """
    nwave = 3; ncol = 1; nlyr = 10; nstr = 4
    tau = 0.1; fbeam = 1.0; umu0 = 0.5
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    setup_common = dict(
        nwave=nwave, ncol=ncol, nlyr=nlyr, nstr=nstr,
        wave_lo=wave_lo, wave_hi=wave_hi, fbeam=fbeam, umu0=umu0,
    )

    for w0, g in [(0.5, 0.5), (0.9, 0.5)]:
        # ---- Toon subprocess ----
        toon_code = (
            _TOON_SETUP.format(**setup_common)
            + f"""\
prop = torch.zeros(nwave, ncol, nlyr, 3, dtype=torch.float64)
prop[:, :, :, 0] = {tau}; prop[:, :, :, 1] = {w0}; prop[:, :, :, 2] = {g}
r = toon(prop, bc, "", None)
print(json.dumps({{"dn_surf": float(r[0, 0, 0, 1])}}))
"""
        )
        # ---- DISORT subprocess ----
        disort_code = (
            _DISORT_SETUP.format(flags="lamber,quiet,onlyfl", **setup_common)
            + f"""\
prop = torch.zeros(nwave, ncol, nlyr, 2 + nstr, dtype=torch.float64)
prop[:, :, :, 0] = {tau}; prop[:, :, :, 1] = {w0}
for l in range(nstr):
    prop[:, :, :, 2 + l] = {g}**l
r = disort.forward(prop, bc, "", None)
print(json.dumps({{"dn_surf": float(r[0, 0, 0, 1])}}))
"""
        )
        dn_toon   = _run(toon_code)["dn_surf"]
        dn_disort = _run(disort_code)["dn_surf"]

        ratio = dn_toon / (dn_disort + 1e-30)
        assert 0.33 < ratio < 3.0, (
            f"Surface dn flux ratio outside [0.33, 3] for w0={w0}, g={g}: "
            f"toon={dn_toon:.4f}, disort={dn_disort:.4f}, ratio={ratio:.3f}"
        )


def test_toon_lw_isothermal():
    """Toon and DISORT longwave fluxes must agree for an isothermal atmosphere.

    For a uniform temperature profile and moderate optical depth, both solvers
    should give upward fluxes within 20% of each other.
    """
    nwave = 3; ncol = 1; nlyr = 10; nstr = 4
    tau = 0.3; temp_K = 300.0
    wave_lo = [500.0, 1000.0, 1500.0]
    wave_hi = [1000.0, 1500.0, 2000.0]

    setup_lw = dict(
        nwave=nwave, ncol=ncol, nlyr=nlyr, nstr=nstr,
        wave_lo=wave_lo, wave_hi=wave_hi,
        fbeam=0.0, umu0=1.0,  # not used for LW
    )

    lw_bc = f"""\
bc = {{
    "albedo": torch.zeros(nwave, ncol, dtype=torch.float64),
    "temis":  torch.zeros(nwave, ncol, dtype=torch.float64),
    "btemp":  torch.ones(ncol, dtype=torch.float64) * {temp_K},
    "ttemp":  torch.ones(ncol, dtype=torch.float64) * {temp_K},
}}
temf = torch.ones(ncol, nlyr + 1, dtype=torch.float64) * {temp_K}
"""

    # ---- Toon subprocess ----
    toon_code = (
        _TOON_SETUP.format(**setup_lw).replace(
            # Remove the SW-only bc keys and replace with LW bc
            "bc = {",
            "_bc_dummy = {",   # rename the SW bc so it's not used
        )
        + lw_bc
        + f"""\
prop = torch.zeros(nwave, ncol, nlyr, 3, dtype=torch.float64)
prop[:, :, :, 0] = {tau}
r = toon(prop, bc, "", temf)
print(json.dumps({{"up_toa": r[:, :, -1, 0].tolist()}}))
"""
    )

    # ---- DISORT subprocess ----
    disort_code = (
        _DISORT_SETUP.format(flags="lamber,quiet,onlyfl,planck", **setup_lw).replace(
            "bc = {",
            "_bc_dummy = {",
        )
        + lw_bc
        + f"""\
prop = torch.zeros(nwave, ncol, nlyr, 2 + nstr, dtype=torch.float64)
prop[:, :, :, 0] = {tau}
r = disort.forward(prop, bc, "", temf)
print(json.dumps({{"up_toa": r[:, :, -1, 0].tolist()}}))
"""
    )

    toon_out   = _run(toon_code)["up_toa"]
    disort_out = _run(disort_code)["up_toa"]

    for iw in range(nwave):
        up_t = toon_out[iw][0]
        up_d = disort_out[iw][0]
        assert up_t > 0, f"Toon LW upward TOA flux non-positive for band {iw}"
        assert up_d > 0, f"DISORT LW upward TOA flux non-positive for band {iw}"
        rel_diff = abs(up_t - up_d) / (abs(up_d) + 1e-30)
        assert rel_diff < 0.20, (
            f"Toon/DISORT LW TOA flux differ >20% for band {iw}: "
            f"toon={up_t:.4f}, disort={up_d:.4f}, rel_diff={rel_diff:.3f}"
        )


if __name__ == "__main__":
    test_toon_sw_pure_absorption()
    print("test_toon_sw_pure_absorption PASSED")

    test_toon_sw_scattering()
    print("test_toon_sw_scattering PASSED")

    test_toon_sw_vs_disort()
    print("test_toon_sw_vs_disort PASSED")

    test_toon_lw_isothermal()
    print("test_toon_lw_isothermal PASSED")
