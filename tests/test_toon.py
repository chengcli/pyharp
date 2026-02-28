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

Requires pyharp >= 1.9.0 and pydisort >= 1.5.0.

Implementation note
-------------------
The pyharp and pydisort packages share PyTorch CPU dispatch stubs.  Creating
multiple solver objects in the same Python process can leave the dispatch
table in an inconsistent state.  Each test case therefore runs in its own
subprocess so each solver gets a fully isolated Python environment.
"""

import json
import math
import subprocess
import sys


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

# Common imports for toon-only subprocesses (no Disort object is created so
# the CPU dispatch table stays in a consistent state).
_TOON_IMPORTS = """\
import json
import torch
import pyharp
from pyharp.cpp import ToonMcKay89 as ToonModule
torch.set_default_dtype(torch.float64)
"""

# Common imports for disort-only subprocesses.
_DISORT_IMPORTS = """\
import json
import torch
import pyharp
import pydisort
torch.set_default_dtype(torch.float64)
"""


def _run_toon(code: str) -> object:
    """Run Toon-only *code* in a fresh subprocess (no Disort object created)."""
    proc = subprocess.run(
        [sys.executable, "-c", _TOON_IMPORTS + code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Subprocess failed (exit {proc.returncode}):\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return json.loads(proc.stdout.strip())


def _run_disort(code: str) -> object:
    """Run Disort-only *code* in a fresh subprocess (no ToonMcKay89 object created)."""
    proc = subprocess.run(
        [sys.executable, "-c", _DISORT_IMPORTS + code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Subprocess failed (exit {proc.returncode}):\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return json.loads(proc.stdout.strip())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_toon_sw_pure_absorption():
    """Toon and DISORT must reproduce the Beer-Lambert law for pure absorption.

    With w0=0 and zero surface albedo, the downward flux at each level follows
    ``fbeam * umu0 * exp(-(nlyr - k) * tau / umu0)`` exactly.  Both solvers
    must match this analytical result and agree with each other.
    """
    nwave = 3; ncol = 1; nlyr = 10; nstr = 4
    tau = 0.1; fbeam = 1000.0; umu0 = 0.5
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    toon_out = _run_toon(f"""\
nwave={nwave}; ncol={ncol}; nlyr={nlyr}
wave_lo={wave_lo}; wave_hi={wave_hi}
opt = pyharp.ToonMcKay89Options()
opt.wave_lower(wave_lo); opt.wave_upper(wave_hi)
toon = ToonModule(opt); toon.double()
bc = {{"fbeam": torch.ones(nwave,ncol,dtype=torch.float64)*{fbeam},
       "umu0":  torch.ones(ncol, dtype=torch.float64)*{umu0},
       "albedo":torch.zeros(nwave,ncol,dtype=torch.float64)}}
prop = torch.zeros(nwave,ncol,nlyr,3,dtype=torch.float64)
prop[:,:,:,0] = {tau}
r = toon(prop, bc, "", None)
print(json.dumps({{"dn": r[0,0,:,1].tolist(), "up": r[0,0,:,0].tolist()}}))
""")

    disort_out = _run_disort(f"""\
nwave={nwave}; ncol={ncol}; nlyr={nlyr}; nstr={nstr}
wave_lo={wave_lo}; wave_hi={wave_hi}
dop = pydisort.DisortOptions()
pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
dop.upward(True); dop.flags("lamber,quiet,onlyfl")
dop.wave_lower(wave_lo); dop.wave_upper(wave_hi)
disort = pydisort.Disort(dop); disort.double()
bc = {{"fbeam": torch.ones(nwave,ncol,dtype=torch.float64)*{fbeam},
       "umu0":  torch.ones(ncol, dtype=torch.float64)*{umu0},
       "albedo":torch.zeros(nwave,ncol,dtype=torch.float64)}}
prop = torch.zeros(nwave,ncol,nlyr,2+nstr,dtype=torch.float64)
prop[:,:,:,0] = {tau}
r = disort.forward(prop, bc, "", None)
print(json.dumps({{"dn": r[0,0,:,1].tolist(), "up": r[0,0,:,0].tolist()}}))
""")

    # Analytical Beer-Lambert law: level k traverses (nlyr-k) layers from TOA
    expected_dn = [
        fbeam * umu0 * math.exp(-(nlyr - k) * tau / umu0)
        for k in range(nlyr + 1)
    ]

    for k in range(nlyr + 1):
        assert abs(toon_out["dn"][k] - expected_dn[k]) < 1e-8, (
            f"Toon Beer-Lambert mismatch at level {k}"
        )
        assert abs(disort_out["dn"][k] - expected_dn[k]) < 1e-8, (
            f"DISORT Beer-Lambert mismatch at level {k}"
        )
        # No upward flux for pure absorption with zero surface albedo
        assert abs(toon_out["up"][k]) < 1e-10, (
            f"Non-zero Toon upward flux at level {k} for pure absorption"
        )
        # Toon and DISORT must agree
        assert abs(toon_out["dn"][k] - disort_out["dn"][k]) < 1e-8, (
            f"Toon/DISORT disagreement at level {k}"
        )


def test_toon_sw_scattering():
    """Toon fluxes must satisfy energy conservation for shortwave scattering.

    This test verifies physical consistency of the two-stream approximation:
      - All fluxes are non-negative.
      - Reflected flux at TOA does not exceed the incident solar flux.
      - Surface downward flux does not exceed the incident solar flux.

    Each (w0, g) case runs in its own subprocess with a single ToonMcKay89
    instance to avoid CPU dispatch state conflicts.
    """
    nwave = 3; ncol = 1; nlyr = 10
    tau = 0.1; fbeam = 1000.0; umu0 = 0.5
    incident = fbeam * umu0
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    for w0, g in [(0.5, 0.0), (0.5, 0.5), (0.9, 0.5)]:
        res = _run_toon(f"""\
nwave={nwave}; ncol={ncol}; nlyr={nlyr}
wave_lo={wave_lo}; wave_hi={wave_hi}
opt = pyharp.ToonMcKay89Options()
opt.wave_lower(wave_lo); opt.wave_upper(wave_hi)
toon = ToonModule(opt); toon.double()
bc = {{"fbeam": torch.ones(nwave,ncol,dtype=torch.float64)*{fbeam},
       "umu0":  torch.ones(ncol, dtype=torch.float64)*{umu0},
       "albedo":torch.zeros(nwave,ncol,dtype=torch.float64)}}
prop = torch.zeros(nwave,ncol,nlyr,3,dtype=torch.float64)
prop[:,:,:,0]={tau}; prop[:,:,:,1]={w0}; prop[:,:,:,2]={g}
r = toon(prop, bc, "", None)
print(json.dumps({{
    "min_up":  float(r[:,:,:,0].min()),
    "min_dn":  float(r[:,:,:,1].min()),
    "up_toa":  float(r[0,0,-1,0]),
    "dn_surf": float(r[0,0,0,1]),
}}))
""")
        assert res["min_up"] >= -1e-10, (
            f"Negative upward flux for w0={w0}, g={g}: {res['min_up']}"
        )
        assert res["min_dn"] >= -1e-10, (
            f"Negative downward flux for w0={w0}, g={g}: {res['min_dn']}"
        )
        assert res["up_toa"] <= incident + 1e-6, (
            f"TOA upward flux exceeds incident for w0={w0}, g={g}"
        )
        assert res["dn_surf"] <= incident + 1e-6, (
            f"Surface downward flux exceeds incident for w0={w0}, g={g}"
        )


def test_toon_sw_vs_disort():
    """Toon and DISORT surface downward fluxes must agree within a factor of 3.

    For optically moderate atmospheres, both solvers give qualitatively
    consistent surface downward fluxes despite the two-stream approximation.
    """
    nwave = 3; ncol = 1; nlyr = 10; nstr = 4
    tau = 0.1; fbeam = 1.0; umu0 = 0.5
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    for w0, g in [(0.5, 0.5), (0.9, 0.5)]:
        toon_out = _run_toon(f"""\
nwave={nwave}; ncol={ncol}; nlyr={nlyr}
wave_lo={wave_lo}; wave_hi={wave_hi}
opt = pyharp.ToonMcKay89Options()
opt.wave_lower(wave_lo); opt.wave_upper(wave_hi)
toon = ToonModule(opt); toon.double()
bc = {{"fbeam": torch.ones(nwave,ncol,dtype=torch.float64)*{fbeam},
       "umu0":  torch.ones(ncol, dtype=torch.float64)*{umu0},
       "albedo":torch.zeros(nwave,ncol,dtype=torch.float64)}}
prop = torch.zeros(nwave,ncol,nlyr,3,dtype=torch.float64)
prop[:,:,:,0]={tau}; prop[:,:,:,1]={w0}; prop[:,:,:,2]={g}
r = toon(prop, bc, "", None)
print(json.dumps({{"dn_surf": float(r[0,0,0,1])}}))
""")
        disort_out = _run_disort(f"""\
nwave={nwave}; ncol={ncol}; nlyr={nlyr}; nstr={nstr}
wave_lo={wave_lo}; wave_hi={wave_hi}
dop = pydisort.DisortOptions()
pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
dop.upward(True); dop.flags("lamber,quiet,onlyfl")
dop.wave_lower(wave_lo); dop.wave_upper(wave_hi)
disort = pydisort.Disort(dop); disort.double()
bc = {{"fbeam": torch.ones(nwave,ncol,dtype=torch.float64)*{fbeam},
       "umu0":  torch.ones(ncol, dtype=torch.float64)*{umu0},
       "albedo":torch.zeros(nwave,ncol,dtype=torch.float64)}}
prop = torch.zeros(nwave,ncol,nlyr,2+nstr,dtype=torch.float64)
prop[:,:,:,0]={tau}; prop[:,:,:,1]={w0}
for l in range(nstr):
    prop[:,:,:,2+l] = {g}**l
r = disort.forward(prop, bc, "", None)
print(json.dumps({{"dn_surf": float(r[0,0,0,1])}}))
""")
        dn_toon   = toon_out["dn_surf"]
        dn_disort = disort_out["dn_surf"]
        ratio = dn_toon / (dn_disort + 1e-30)
        assert 0.33 < ratio < 3.0, (
            f"Surface dn flux ratio outside [0.33, 3] for w0={w0}, g={g}: "
            f"toon={dn_toon:.4f}, disort={dn_disort:.4f}, ratio={ratio:.3f}"
        )


def test_toon_lw_isothermal():
    """Toon and DISORT longwave fluxes must agree for an isothermal atmosphere.

    For a uniform temperature profile and moderate optical depth, the upward
    flux at TOA from Toon must be within 20% of the DISORT reference.
    """
    nwave = 3; ncol = 1; nlyr = 10; nstr = 4
    tau = 0.3; temp_K = 300.0
    wave_lo = [500.0, 1000.0, 1500.0]
    wave_hi = [1000.0, 1500.0, 2000.0]

    lw_bc = f"""\
nwave={nwave}; ncol={ncol}; nlyr={nlyr}; nstr={nstr}
wave_lo={wave_lo}; wave_hi={wave_hi}; temp_K={temp_K}
bc = {{"albedo": torch.zeros(nwave,ncol,dtype=torch.float64),
       "temis":  torch.zeros(nwave,ncol,dtype=torch.float64),
       "btemp":  torch.ones(ncol, dtype=torch.float64)*temp_K,
       "ttemp":  torch.ones(ncol, dtype=torch.float64)*temp_K}}
temf = torch.ones(ncol,nlyr+1,dtype=torch.float64)*temp_K
"""

    toon_out = _run_toon(lw_bc + f"""\
opt = pyharp.ToonMcKay89Options()
opt.wave_lower(wave_lo); opt.wave_upper(wave_hi)
toon = ToonModule(opt); toon.double()
prop = torch.zeros(nwave,ncol,nlyr,3,dtype=torch.float64)
prop[:,:,:,0] = {tau}
r = toon(prop, bc, "", temf)
print(json.dumps({{"up_toa": r[:,:,-1,0].tolist()}}))
""")

    disort_out = _run_disort(lw_bc + f"""\
dop = pydisort.DisortOptions()
pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
dop.upward(True); dop.flags("lamber,quiet,onlyfl,planck")
dop.wave_lower(wave_lo); dop.wave_upper(wave_hi)
disort = pydisort.Disort(dop); disort.double()
prop = torch.zeros(nwave,ncol,nlyr,2+nstr,dtype=torch.float64)
prop[:,:,:,0] = {tau}
r = disort.forward(prop, bc, "", temf)
print(json.dumps({{"up_toa": r[:,:,-1,0].tolist()}}))
""")

    for iw in range(nwave):
        up_t = toon_out["up_toa"][iw][0]
        up_d = disort_out["up_toa"][iw][0]
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
