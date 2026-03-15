#!/usr/bin/env python3
"""
Validate Toon's two-stream solver against PICASO.

This script comprehensively compares PyHarp's ToonMcKay89 solver with PICASO's
two-stream implementation for both shortwave (reflected) and longwave (thermal) cases.

PICASO is used as the reference implementation. Any significant discrepancies indicate
potential issues in PyHarp's Toon solver that need to be corrected.
"""

import numpy as np
import torch
import pyharp
from picaso import fluxes as picaso_fluxes

torch.set_default_dtype(torch.float64)


def setup_test_atmosphere():
    """Create a simple test atmosphere for validation."""
    nlyr = 20
    nlevel = nlyr + 1

    # Optical properties
    tau_per_layer = 0.1
    w0 = 0.8  # single scattering albedo
    g = 0.5   # asymmetry parameter

    # For PICASO (layer properties)
    dtau = np.ones(nlyr) * tau_per_layer
    tau = np.cumsum(dtau)
    tau_levels = np.concatenate([[0], tau])
    w0_arr = np.ones(nlyr) * w0
    cosb = np.ones(nlyr) * g

    # Rayleigh parameters (no separate rayleigh in this test)
    gcos2 = np.zeros(nlyr)
    ftau_cld = np.ones(nlyr)
    ftau_ray = np.zeros(nlyr)

    return {
        'nlyr': nlyr,
        'nlevel': nlevel,
        'dtau': dtau,
        'tau': tau,
        'tau_levels': tau_levels,
        'w0': w0_arr,
        'cosb': cosb,
        'gcos2': gcos2,
        'ftau_cld': ftau_cld,
        'ftau_ray': ftau_ray
    }


def validate_shortwave():
    """Compare Toon shortwave (reflected) with PICASO."""
    print("=" * 80)
    print("SHORTWAVE (REFLECTED) VALIDATION")
    print("=" * 80)

    atm = setup_test_atmosphere()
    nlyr = atm['nlyr']
    nlevel = atm['nlevel']

    # Wavelength/wavenumber setup
    nwave = 1
    wave_lo = [500.0]  # cm^-1
    wave_hi = [1000.0]
    wno = np.array([(wave_lo[0] + wave_hi[0]) / 2.0])
    dwno = np.array([wave_hi[0] - wave_lo[0]])

    # Solar parameters
    fbeam = 1000.0  # incident solar flux
    umu0 = 0.5      # cos(solar zenith angle)
    albedo = 0.1    # surface albedo

    # -------------------------------------------------------------------------
    # PyHarp Toon solver
    # -------------------------------------------------------------------------
    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    # Prepare PyHarp input: shape (nwave, ncol, nlyr, nprop)
    ncol = 1
    nprop = 3
    prop = torch.zeros(nwave, ncol, nlyr, nprop)
    prop[:, :, :, 0] = torch.tensor(atm['dtau']).unsqueeze(0).unsqueeze(0)  # tau
    prop[:, :, :, 1] = torch.tensor(atm['w0']).unsqueeze(0).unsqueeze(0)    # w0
    prop[:, :, :, 2] = torch.tensor(atm['cosb']).unsqueeze(0).unsqueeze(0)  # g

    bc = {
        "fbeam": torch.ones(nwave, ncol) * fbeam,
        "umu0": torch.ones(ncol) * umu0,
        "albedo": torch.ones(nwave, ncol) * albedo
    }

    result_toon = toon(prop, **bc)

    # Extract fluxes: shape is (nwave, ncol, nlevel, 2)
    # [..., 0] = upward, [..., 1] = downward
    flux_up_toon = result_toon[0, 0, :, 0].numpy()
    flux_dn_toon = result_toon[0, 0, :, 1].numpy()

    # -------------------------------------------------------------------------
    # PICASO solver
    # -------------------------------------------------------------------------
    # PICASO parameters
    numg = 1  # number of gauss angles
    numt = 1  # number of chebyshev angles
    nwno = 1

    # PICASO needs these arrays expanded to (nlayer, nwave)
    dtau_picaso = atm['dtau'][:, np.newaxis]
    tau_picaso = atm['tau'][:, np.newaxis]
    w0_picaso = atm['w0'][:, np.newaxis]
    cosb_picaso = atm['cosb'][:, np.newaxis]
    gcos2_picaso = atm['gcos2'][:, np.newaxis]
    ftau_cld_picaso = atm['ftau_cld'][:, np.newaxis]
    ftau_ray_picaso = atm['ftau_ray'][:, np.newaxis]

    # For delta-eddington (not applied in this test)
    dtau_og = dtau_picaso.copy()
    tau_og = tau_picaso.copy()
    w0_og = w0_picaso.copy()
    cosb_og = cosb_picaso.copy()

    # Surface reflectance
    surf_reflect = np.ones(nwno) * albedo

    # Incident angles (for disk integration)
    ubar0 = np.array([[umu0]])  # incident
    ubar1 = np.array([[0.5]])   # outgoing (for 2-stream)
    cos_theta = np.array([[1.0]])

    # Solar flux
    F0PI = np.ones(nwno) * fbeam * umu0

    # Phase function parameters (for single/multiple scattering)
    single_phase = np.zeros(nwno)
    multi_phase = np.zeros(nwno)
    frac_a = np.zeros(nwno)
    frac_b = np.zeros(nwno)
    frac_c = np.zeros(nwno)
    constant_back = np.zeros(nwno)
    constant_forward = np.zeros(nwno)

    # Call PICASO
    result_picaso = picaso_fluxes.get_reflected_1d(
        nlevel, wno, nwno, numg, numt,
        dtau_picaso, tau_picaso, w0_picaso, cosb_picaso,
        gcos2_picaso, ftau_cld_picaso, ftau_ray_picaso,
        dtau_og, tau_og, w0_og, cosb_og,
        surf_reflect, ubar0, ubar1, cos_theta, F0PI,
        single_phase, multi_phase,
        frac_a, frac_b, frac_c, constant_back, constant_forward,
        get_toa_intensity=0, get_lvl_flux=1,
        toon_coefficients=0, b_top=0
    )

    # PICASO returns: (flux_minus, flux_plus, ...)
    # flux_minus = downward, flux_plus = upward
    # shape: (numg, numt, nwno, nlevel)
    flux_dn_picaso = result_picaso[0][0, 0, 0, :]
    flux_up_picaso = result_picaso[1][0, 0, 0, :]

    # -------------------------------------------------------------------------
    # Compare results
    # -------------------------------------------------------------------------
    print("\nShortwave Flux Comparison:")
    print("-" * 80)
    print(f"{'Level':>5} {'Toon Up':>12} {'PICASO Up':>12} {'Rel Diff':>12} "
          f"{'Toon Dn':>12} {'PICASO Dn':>12} {'Rel Diff':>12}")
    print("-" * 80)

    max_rel_diff_up = 0.0
    max_rel_diff_dn = 0.0

    for i in range(nlevel):
        rel_diff_up = abs(flux_up_toon[i] - flux_up_picaso[i]) / (abs(flux_up_picaso[i]) + 1e-30)
        rel_diff_dn = abs(flux_dn_toon[i] - flux_dn_picaso[i]) / (abs(flux_dn_picaso[i]) + 1e-30)

        max_rel_diff_up = max(max_rel_diff_up, rel_diff_up)
        max_rel_diff_dn = max(max_rel_diff_dn, rel_diff_dn)

        print(f"{i:5d} {flux_up_toon[i]:12.6f} {flux_up_picaso[i]:12.6f} {rel_diff_up:12.6f} "
              f"{flux_dn_toon[i]:12.6f} {flux_dn_picaso[i]:12.6f} {rel_diff_dn:12.6f}")

    print("-" * 80)
    print(f"Maximum relative difference (upward):   {max_rel_diff_up:.6f}")
    print(f"Maximum relative difference (downward): {max_rel_diff_dn:.6f}")

    # Check if differences are acceptable (< 5%)
    tolerance = 0.05
    if max_rel_diff_up > tolerance or max_rel_diff_dn > tolerance:
        print(f"\n⚠️  WARNING: Differences exceed tolerance of {tolerance*100:.1f}%")
        print("    Toon solver may need corrections!")
        return False
    else:
        print(f"\n✓ PASSED: All differences within {tolerance*100:.1f}% tolerance")
        return True


def validate_longwave():
    """Compare Toon longwave (thermal) with PICASO."""
    print("\n" + "=" * 80)
    print("LONGWAVE (THERMAL) VALIDATION")
    print("=" * 80)

    atm = setup_test_atmosphere()
    nlyr = atm['nlyr']
    nlevel = atm['nlevel']

    # Wavelength/wavenumber setup
    nwave = 1
    wave_lo = [500.0]  # cm^-1
    wave_hi = [1000.0]
    wno = np.array([(wave_lo[0] + wave_hi[0]) / 2.0])
    dwno = np.array([wave_hi[0] - wave_lo[0]])

    # Temperature profile (isothermal)
    temp_K = 300.0
    tlevel = np.ones(nlevel) * temp_K

    # Pressure profile (for PICASO)
    ptop = 1e4   # dyne/cm^2 (0.01 bar)
    pbot = 1e7   # dyne/cm^2 (10 bar)
    plevel = np.logspace(np.log10(ptop), np.log10(pbot), nlevel)

    # Surface properties
    surf_emis = 1.0
    albedo = 0.0  # no reflection in thermal

    # -------------------------------------------------------------------------
    # PyHarp Toon solver
    # -------------------------------------------------------------------------
    opt = pyharp.ToonMcKay89Options()
    opt.wave_lower(wave_lo)
    opt.wave_upper(wave_hi)
    toon = pyharp.ToonMcKay89(opt)

    # Prepare PyHarp input: shape (nwave, ncol, nlyr, nprop)
    ncol = 1
    nprop = 3
    prop = torch.zeros(nwave, ncol, nlyr, nprop)
    prop[:, :, :, 0] = torch.tensor(atm['dtau']).unsqueeze(0).unsqueeze(0)  # tau
    prop[:, :, :, 1] = torch.tensor(atm['w0']).unsqueeze(0).unsqueeze(0)    # w0
    prop[:, :, :, 2] = torch.tensor(atm['cosb']).unsqueeze(0).unsqueeze(0)  # g

    bc = {
        "albedo": torch.ones(nwave, ncol) * albedo,
        "temis": torch.ones(nwave, ncol) * surf_emis,
        "btemp": torch.ones(ncol) * temp_K,
        "ttemp": torch.ones(ncol) * temp_K
    }

    temf = torch.tensor(tlevel).unsqueeze(0)  # (ncol, nlevel)

    result_toon = toon(prop, temf=temf, **bc)

    # Extract fluxes
    flux_up_toon = result_toon[0, 0, :, 0].numpy()
    flux_dn_toon = result_toon[0, 0, :, 1].numpy()

    # -------------------------------------------------------------------------
    # PICASO solver
    # -------------------------------------------------------------------------
    numg = 1
    numt = 1
    nwno = 1

    # PICASO needs these arrays expanded to (nlayer, nwave)
    dtau_picaso = atm['dtau'][:, np.newaxis]
    w0_picaso = atm['w0'][:, np.newaxis]
    cosb_picaso = atm['cosb'][:, np.newaxis]

    # Surface parameters
    surf_reflect = np.ones(nwno) * albedo
    hard_surface = 1  # use surface boundary condition

    # Outgoing angles
    ubar1 = np.array([[0.5]])

    # Calculation type
    calc_type = 0  # 0 = regular thermal

    # Call PICASO
    result_picaso = picaso_fluxes.get_thermal_1d(
        nlevel, wno, nwno, numg, numt, tlevel,
        dtau_picaso, w0_picaso, cosb_picaso,
        plevel, ubar1, surf_reflect, hard_surface, dwno, calc_type
    )

    # PICASO returns: (flux_minus, flux_plus, ...)
    # shape: (numg, numt, nwno, nlevel)
    flux_dn_picaso = result_picaso[0][0, 0, 0, :]
    flux_up_picaso = result_picaso[1][0, 0, 0, :]

    # -------------------------------------------------------------------------
    # Compare results
    # -------------------------------------------------------------------------
    print("\nLongwave Flux Comparison:")
    print("-" * 80)
    print(f"{'Level':>5} {'Toon Up':>12} {'PICASO Up':>12} {'Rel Diff':>12} "
          f"{'Toon Dn':>12} {'PICASO Dn':>12} {'Rel Diff':>12}")
    print("-" * 80)

    max_rel_diff_up = 0.0
    max_rel_diff_dn = 0.0

    for i in range(nlevel):
        rel_diff_up = abs(flux_up_toon[i] - flux_up_picaso[i]) / (abs(flux_up_picaso[i]) + 1e-30)
        rel_diff_dn = abs(flux_dn_toon[i] - flux_dn_picaso[i]) / (abs(flux_dn_picaso[i]) + 1e-30)

        max_rel_diff_up = max(max_rel_diff_up, rel_diff_up)
        max_rel_diff_dn = max(max_rel_diff_dn, rel_diff_dn)

        print(f"{i:5d} {flux_up_toon[i]:12.6f} {flux_up_picaso[i]:12.6f} {rel_diff_up:12.6f} "
              f"{flux_dn_toon[i]:12.6f} {flux_dn_picaso[i]:12.6f} {rel_diff_dn:12.6f}")

    print("-" * 80)
    print(f"Maximum relative difference (upward):   {max_rel_diff_up:.6f}")
    print(f"Maximum relative difference (downward): {max_rel_diff_dn:.6f}")

    # Check if differences are acceptable (< 5%)
    tolerance = 0.05
    if max_rel_diff_up > tolerance or max_rel_diff_dn > tolerance:
        print(f"\n⚠️  WARNING: Differences exceed tolerance of {tolerance*100:.1f}%")
        print("    Toon solver may need corrections!")
        return False
    else:
        print(f"\n✓ PASSED: All differences within {tolerance*100:.1f}% tolerance")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TOON vs PICASO VALIDATION SUITE")
    print("=" * 80)
    print("\nThis script compares PyHarp's Toon-McKay89 solver with PICASO")
    print("PICASO is used as the reference implementation")
    print("=" * 80)

    sw_passed = validate_shortwave()
    lw_passed = validate_longwave()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Shortwave validation: {'PASSED ✓' if sw_passed else 'FAILED ✗'}")
    print(f"Longwave validation:  {'PASSED ✓' if lw_passed else 'FAILED ✗'}")

    if sw_passed and lw_passed:
        print("\n✓ All validations passed!")
        exit(0)
    else:
        print("\n✗ Some validations failed - Toon solver needs corrections")
        exit(1)
