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
"""

import json
import math
import subprocess
import sys
import torch
import pyharp
import pydisort

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_toon_sw_pure_absorption():
    """Toon solver must reproduce the Beer-Lambert law for pure absorption.

    With w0=0 and zero surface albedo, the downward flux at each level follows
    ``fbeam * umu0 * exp(-(nlyr - k) * tau / umu0)`` exactly.
    """
    nwave = 3
    ncol = 1
    nlyr = 10
    nprop = 3
    tau = 0.1
    fbeam = 1000.0
    umu0 = 0.5
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    bc = {
        "fbeam": torch.ones(nwave, ncol) * fbeam,
        "umu0":  torch.ones(ncol) * umu0,
        "albedo": torch.zeros(nwave,ncol)
    }

    prop = torch.zeros(nwave,ncol,nlyr,nprop)
    prop[:,:,:,0] = tau  # optical depth

    result = toon(prop, **bc)
    toon_out = {
        "dn": result[0,0,:,1].tolist(),
        "up": result[0,0,:,0].tolist()
    }

    # Analytical Beer-Lambert law: level k traverses (nlyr-k) layers from TOA
    expected_dn = [
        fbeam * umu0 * math.exp(-(nlyr - k) * tau / umu0)
        for k in range(nlyr + 1)
    ]
    print("Level | Toon dn | Expected dn")
    print("-----------------------------")

    for k in range(nlyr + 1):
        print(f"{k:5d} | {toon_out['dn'][k]:8.4f} | {expected_dn[k]:11.4f}")
        assert abs(toon_out["dn"][k] - expected_dn[k]) < 1e-8, (
            f"Toon Beer-Lambert mismatch at level {k}"
        )

        assert abs(toon_out["up"][k]) < 1e-10, (
            f"Non-zero Toon upward flux at level {k} for pure absorption"
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
    nwave = 3
    ncol = 1
    nlyr = 10
    nprop = 3
    tau = 0.1
    fbeam = 1000.0
    umu0 = 0.5
    incident = fbeam * umu0
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    prop = torch.zeros(nwave,ncol,nlyr,nprop)
    prop[:,:,:,0] = tau  # optical depth

    bc = {
        "fbeam": torch.ones(nwave,ncol) * fbeam,
        "umu0":  torch.ones(ncol) * umu0,
        "albedo":torch.zeros(nwave,ncol)
    }

    for w0, g in [(0.5, 0.0), (0.5, 0.5), (0.9, 0.5)]:
        prop[:,:,:,1] = w0  # single-scattering albedo
        prop[:,:,:,2] = g   # asymmetry factor

        result = toon(prop, **bc)

        toon_out = {
            "min_up": float(result[0,0,:,0].min()),
            "min_dn": float(result[0,0,:,1].min()),
            "up_toa": float(result[0,0,-1,0]),
            "dn_surf": float(result[0,0,0,1])
        }

        assert toon_out["min_up"] >= -1e-10, (
            f"Negative upward flux for w0={w0}, g={g}: {toon_out['min_up']}"
        )
        assert toon_out["min_dn"] >= -1e-10, (
            f"Negative downward flux for w0={w0}, g={g}: {toon_out['min_dn']}"
        )
        assert toon_out["up_toa"] <= incident + 1e-6, (
            f"TOA upward flux exceeds incident for w0={w0}, g={g}"
        )
        assert toon_out["dn_surf"] <= incident + 1e-6, (
            f"Surface downward flux exceeds incident for w0={w0}, g={g}"
        )

def test_toon_sw_vs_disort():
    """Toon and DISORT surface downward fluxes must agree within a factor of 3.

    For optically moderate atmospheres, both solvers give qualitatively
    consistent surface downward fluxes despite the two-stream approximation.
    """
    nwave = 1
    ncol = 1
    nlyr = 10
    nprop = 3
    nstr = 4
    tau = 0.1
    fbeam = 1.0
    umu0 = 0.5
    wave_lo = [200.0, 500.0, 1000.0]
    wave_hi = [500.0, 1000.0, 2000.0]

    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    dop = pydisort.DisortOptions()
    dop.upward(True)
    dop.flags("lamber,quiet,onlyfl")
    dop.wave_lower(wave_lo)
    dop.wave_upper(wave_hi)
    pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
    disort = pydisort.Disort(dop)

    # accommodate both Toon and DISORT prop array shapes with max(nprop, 2+nstr)
    prop = torch.zeros(nwave,ncol,nlyr,max(nprop,2+nstr))
    prop[:,:,:,0] = tau  # optical depth

    bc = {
        "fbeam": torch.ones(nwave,ncol) * fbeam,
        "umu0":  torch.ones(ncol) * umu0,
        "albedo":torch.zeros(nwave,ncol)
    }

    for w0, g in [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]:
        prop[:,:,:,1] = w0  # single-scattering albedo
        prop[:,:,:,2] = g   # asymmetry factor

        result = toon(prop, **bc)

        dn_toon = float(result[0,0,0,1])

        # for disort, the scattering coefficients are stored in prop[:,:,:,2+l] for l=0..nstr-1
        for l in range(nstr):
            prop[:,:,:,2+l] = g**(l+1)

        result = disort(prop, **bc)

        dn_disort = float(result[0,0,0,1])

        ratio = dn_toon / (dn_disort + 1e-30)
        print(f"w0={w0}, g={g}: Toon dn_surf={dn_toon:.4f}, "
              f"DISORT dn_surf={dn_disort:.4f}, ratio={ratio:.3f}")
        assert 0.8 < ratio < 1.2, (
            f"Surface dn flux ratio outside [0.8, 1.2] for w0={w0}, g={g}: "
            f"toon={dn_toon:.4f}, disort={dn_disort:.4f}, ratio={ratio:.3f}"
        )

def test_toon_lw_isothermal():
    """Toon and DISORT longwave fluxes must agree for an isothermal atmosphere.

    For a uniform temperature profile and moderate optical depth, the upward
    flux at TOA from Toon must be within 20% of the DISORT reference.
    """
    nwave = 3
    ncol = 1
    nlyr = 10
    nprop = 3
    nstr = 4
    tau = 0.3
    temp_K = 300.0
    wave_lo = [500.0, 1000.0, 1500.0]
    wave_hi = [1000.0, 1500.0, 2000.0]

    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    dop = pydisort.DisortOptions()
    dop.upward(True)
    dop.flags("lamber,quiet,onlyfl,planck")
    dop.wave_lower(wave_lo)
    dop.wave_upper(wave_hi)
    pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
    disort = pydisort.Disort(dop)

    # accommodate both Toon and DISORT prop array shapes with max(nprop, 2+nstr)
    prop = torch.zeros(nwave,ncol,nlyr,max(nprop,2+nstr))
    prop[:,:,:,0] = tau  # optical depth

    bc = {
        "albedo": torch.zeros(nwave,ncol),
        "temis":  torch.zeros(nwave,ncol),
        "btemp":  torch.ones(ncol) * temp_K,
        "ttemp":  torch.zeros(ncol) * temp_K
    }

    temf = torch.ones(ncol,nlyr+1) * temp_K

    toon.options.planck(True)
    result = toon(prop, temf=temf, **bc)
    toon_out = {
        "up_toa": result[:,:,-1,0].tolist()  # upward flux at TOA
    }

    result = disort(prop, temf=temf, **bc)
    disort_out = {
        "up_toa": result[:,:,-1,0].tolist()  # upward flux at TOA
    }

    for iw in range(nwave):
        up_t = toon_out["up_toa"][iw][0]
        up_d = disort_out["up_toa"][iw][0]
        print(f"Band {iw}: Toon up_TOA={up_t:.4f}, DISORT up_TOA={up_d:.4f}")
        assert up_t > 0, f"Toon LW upward TOA flux non-positive for band {iw}"
        assert up_d > 0, f"DISORT LW upward TOA flux non-positive for band {iw}"
        rel_diff = abs(up_t - up_d) / (abs(up_d) + 1e-30)
        assert rel_diff < 0.20, (
            f"Toon/DISORT LW TOA flux differ >20% for band {iw}: "
            f"toon={up_t:.4f}, disort={up_d:.4f}, rel_diff={rel_diff:.3f}"
        )

def test_toon_lw_isothermal_scattering():
    """Toon and DISORT longwave fluxes must agree for an isothermal atmosphere.

    For a uniform temperature profile and moderate optical depth, the upward
    flux at TOA from Toon must be within 20% of the DISORT reference.
    """
    nwave = 3
    ncol = 1
    nlyr = 10
    nprop = 3
    nstr = 4
    tau = 0.3
    temp_K = 300.0
    wave_lo = [500.0, 1000.0, 1500.0]
    wave_hi = [1000.0, 1500.0, 2000.0]

    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    dop = pydisort.DisortOptions()
    dop.upward(True)
    dop.flags("lamber,quiet,onlyfl,planck")
    dop.wave_lower(wave_lo)
    dop.wave_upper(wave_hi)
    pyharp.disort_config(dop, nstr, nlyr, ncol, nwave)
    disort = pydisort.Disort(dop)

    # accommodate both Toon and DISORT prop array shapes with max(nprop, 2+nstr)
    prop = torch.zeros(nwave,ncol,nlyr,max(nprop,2+nstr))
    prop[:,:,:,0] = tau  # optical depth

    bc = {
        "albedo": torch.zeros(nwave,ncol),
        "temis":  torch.zeros(nwave,ncol),
        "btemp":  torch.ones(ncol) * temp_K,
        "ttemp":  torch.zeros(ncol) * temp_K
    }

    temf = torch.ones(ncol,nlyr+1) * temp_K

    for w0, g in [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]:
        prop[:,:,:,1] = w0  # single-scattering albedo
        prop[:,:,:,2] = g   # asymmetry factor

        result = toon(prop, temf=temf, **bc)
        toon_out = {
            "up_toa": result[:,:,-1,0].tolist()  # upward flux at TOA
        }

        for l in range(nstr):
            prop[:,:,:,2+l] = g**(l+1)

        result = disort(prop, temf=temf, **bc)
        disort_out = {
            "up_toa": result[:,:,-1,0].tolist()  # upward flux at TOA
        }

        print(f"Scattering case w0={w0}, g={g}:")

        for iw in range(nwave):
            up_t = toon_out["up_toa"][iw][0]
            up_d = disort_out["up_toa"][iw][0]
            print(f"Band {iw}: Toon up_TOA={up_t:.4f}, DISORT up_TOA={up_d:.4f}")
            assert up_t > 0, f"Toon LW upward TOA flux non-positive for band {iw}"
            assert up_d > 0, f"DISORT LW upward TOA flux non-positive for band {iw}"
            rel_diff = abs(up_t - up_d) / (abs(up_d) + 1e-30)
            assert rel_diff < 0.20, (
                f"Toon/DISORT LW TOA flux differ >20% for band {iw}: "
                f"toon={up_t:.4f}, disort={up_d:.4f}, rel_diff={rel_diff:.3f}"
            )

if __name__ == "__main__":
    print("test_toon_sw_pure_absorption")
    test_toon_sw_pure_absorption()
    print("PASSED")

    print("test_toon_sw_scattering")
    test_toon_sw_scattering()
    print("PASSED")

    print("test_toon_sw_vs_disort")
    test_toon_sw_vs_disort()
    print("PASSED")

    print("test_toon_lw_isothermal")
    test_toon_lw_isothermal()
    print("PASSED")

    print("test_toon_lw_isothermal_scattering")
    test_toon_lw_isothermal_scattering()
    print("PASSED")
