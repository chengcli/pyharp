#pragma once

// C/C++
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// base
#include <configure.h>

// harp
#include <harp/utils/alloc.h>

#include "dtridgl_impl.h"

#define DTAU_IN(i) prop[(nlay - (i) - 1) * len1]
#define W_IN(i) prop[(nlay - (i) - 1) * len1 + 1]
#define G_IN(i) prop[(nlay - (i) - 1) * len1 + 2]
#define BE_IN(i) be[nlev - (i) - 1]
// 4 channels per level: [0]=level up, [1]=level down,
// [2]=layer-midpoint up, [3]=layer-midpoint down.
#define FLX_UP(i) flx[4 * (nlev - (i) - 1)]
#define FLX_DN(i) flx[4 * (nlev - (i) - 1) + 1]
// Midpoint flux of internal LAYER k (between internal levels k and k+1) is
// stored in channels [2,3] of the layer's upper-level slot (internal level k).
// The deepest external slot (internal level nlay) holds no layer midpoint -> 0.
#define FLX_UP_MID(k) flx[4 * (nlev - (k) - 1) + 2]
#define FLX_DN_MID(k) flx[4 * (nlev - (k) - 1) + 3]

namespace harp {

template <typename T>
DISPATCH_MACRO T longwave_exprel(T x) {
  return x == static_cast<T>(0) ? static_cast<T>(1) : expm1(x) / x;
}

template <typename T>
DISPATCH_MACRO T longwave_down_resonant_ratio(T delta, T a) {
  T const b = (static_cast<T>(1) + delta) * a;
  return a * exp(-fmin(a, b)) * longwave_exprel(-fabs(delta * a));
}

// The same, given exp(-a) already evaluated by the caller: it is reused
// whenever fmin picks a, which is half of the angles in a weakly scattering
// layer.
template <typename T>
DISPATCH_MACRO T longwave_down_resonant_ratio(T delta, T a, T exp_neg_a) {
  T const b = (static_cast<T>(1) + delta) * a;
  T const m = fmin(a, b);
  return a * (m == a ? exp_neg_a : exp(-m)) * longwave_exprel(-fabs(delta * a));
}

template <typename T>
DISPATCH_MACRO T longwave_up_resonant_ratio(T delta, T a) {
  return a * longwave_exprel(delta * a);
}

template <typename T>
DISPATCH_MACRO T longwave_midpoint_up_resonant_ratio(T delta, T half_a) {
  return exp((static_cast<T>(1) + delta) * half_a) * half_a *
         longwave_exprel(delta * half_a);
}

template <typename T>
DISPATCH_MACRO void toon_mckay89_longwave(int nlay, const T* be, const T* prop,
                                          T a_surf_in, T bsurf_in,
                                          int top_emission_flag,
                                          T btop_factor, bool hard_surface,
                                          bool delta_eddington_lw, T* flx,
                                          int len1, char* work,
                                          int wstride = 1) {
  int nlev = nlay + 1;
  int l = 2 * nlay;
  int lm2 = l - 2;
  int lm1 = l - 1;

  // Constants
  const int nmu = 5;
  const T twopi = 2.0 * M_PI;
  const T ubari = 0.5;

  const T uarr[] = {0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821,
                    0.9601901429};
  const T wuarr[] = {0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381,
                     0.0967815902};

  // --- Work Variables Allocation ---
  auto dtau = alloc_strided<T>(work, nlay, wstride);
  auto tau = alloc_strided<T>(work, nlev, wstride);
  auto w0 = alloc_strided<T>(work, nlay, wstride);
  auto hg = alloc_strided<T>(work, nlay, wstride);
  auto B0 = alloc_strided<T>(work, nlay, wstride);
  auto B1 = alloc_strided<T>(work, nlay, wstride);
  auto lam = alloc_strided<T>(work, nlay, wstride);
  auto gam = alloc_strided<T>(work, nlay, wstride);
  auto alp = alloc_strided<T>(work, nlay, wstride);
  auto term = alloc_strided<T>(work, nlay, wstride);

  auto Cpm1 = alloc_strided<T>(work, nlay, wstride);
  auto Cmm1 = alloc_strided<T>(work, nlay, wstride);
  auto Cp = alloc_strided<T>(work, nlay, wstride);
  auto Cm = alloc_strided<T>(work, nlay, wstride);

  auto Ep = alloc_strided<T>(work, nlay, wstride);
  auto Em = alloc_strided<T>(work, nlay, wstride);
  auto E1 = alloc_strided<T>(work, nlay, wstride);
  auto E2 = alloc_strided<T>(work, nlay, wstride);
  auto E3 = alloc_strided<T>(work, nlay, wstride);
  auto E4 = alloc_strided<T>(work, nlay, wstride);

  auto Af = alloc_strided<T>(work, l, wstride);
  auto Bf = alloc_strided<T>(work, l, wstride);
  auto Cf = alloc_strided<T>(work, l, wstride);
  auto Df = alloc_strided<T>(work, l, wstride);
  auto xkk = alloc_strided<T>(work, l, wstride);
  auto xk1 = alloc_strided<T>(work, nlay, wstride);
  auto xk2 = alloc_strided<T>(work, nlay, wstride);

  auto g = alloc_strided<T>(work, nlay, wstride);
  auto h = alloc_strided<T>(work, nlay, wstride);
  auto xj = alloc_strided<T>(work, nlay, wstride);
  auto xk = alloc_strided<T>(work, nlay, wstride);
  auto alpha1 = alloc_strided<T>(work, nlay, wstride);
  auto alpha2 = alloc_strided<T>(work, nlay, wstride);
  auto sigma1 = alloc_strided<T>(work, nlay, wstride);
  auto sigma2 = alloc_strided<T>(work, nlay, wstride);

  auto em1 = alloc_strided<T>(work, nlay, wstride);
  auto Ep_mid = alloc_strided<T>(work, nlay, wstride);
  auto em1_mid = alloc_strided<T>(work, nlay, wstride);
  // Angle-summed fluxes at the levels and the layer midpoints. The sweeps
  // accumulate here and write flx once, rather than adding every angle into
  // the output, whose columns sit a whole profile apart in memory.
  auto facc = alloc_strided<T>(work, nlev, wstride);
  auto facc_mid = alloc_strided<T>(work, nlay, wstride);

  // Delta-Eddington scaling for longwave: configurable.
  // When enabled (delta_eddington_lw=true), rescale w0, dtau, g as in FMS.
  // When disabled (default), use raw values as in PICASO.
  for (int i = 0; i < nlay; i++) {
    if (delta_eddington_lw) {
      T g_sq = G_IN(i) * G_IN(i);
      w0[i] = ((1.0 - g_sq) * W_IN(i)) / (1.0 - W_IN(i) * g_sq);
      dtau[i] = (1.0 - W_IN(i) * g_sq) * DTAU_IN(i);
      hg[i] = G_IN(i) / (1.0 + G_IN(i));
    } else {
      w0[i] = W_IN(i);
      dtau[i] = DTAU_IN(i);
      hg[i] = G_IN(i);
    }
  }

  tau[0] = 0.0;
  for (int k = 0; k < nlay; k++) {
    tau[k + 1] = tau[k] + dtau[k];
  }

  /* debug: print dtau_in and tau_in
  for (int i = 0; i < nlay; i++) {
    printf("Layer %d: DTAU_IN = %e, tau_in = %e, be = %e\n", i, DTAU_IN(i),
  tau[i + 1], be[i]);
  }*/

  for (int k = 0; k < nlay; k++) {
    alp[k] = sqrt((1.0 - w0[k]) / (1.0 - w0[k] * hg[k]));
    lam[k] = alp[k] * (1.0 - w0[k] * hg[k]) / ubari;
    gam[k] = (1.0 - alp[k]) / (1.0 + alp[k]);
    term[k] = ubari / (1.0 - w0[k] * hg[k]);

    if (dtau[k] <= 1.0e-6) {
      B1[k] = 0.0;
      B0[k] = 0.5 * (BE_IN(k + 1) + BE_IN(k));
    } else {
      B1[k] = (BE_IN(k + 1) - BE_IN(k)) / dtau[k];
      B0[k] = BE_IN(k);
    }

    Cpm1[k] = B0[k] + B1[k] * term[k];
    Cmm1[k] = B0[k] - B1[k] * term[k];
    Cp[k] = B0[k] + B1[k] * dtau[k] + B1[k] * term[k];
    Cm[k] = B0[k] + B1[k] * dtau[k] - B1[k] * term[k];

    T exptrm = fmin(lam[k] * dtau[k], 35.0);
    Ep[k] = exp(exptrm);
    Em[k] = 1.0 / Ep[k];
    E1[k] = Ep[k] + gam[k] * Em[k];
    E2[k] = Ep[k] - gam[k] * Em[k];
    E3[k] = gam[k] * Ep[k] + Em[k];
    E4[k] = gam[k] * Ep[k] - Em[k];
  }

  // Top boundary: controlled by top_emission_flag
  // top_emission_flag = 0: no incoming radiation at TOA (Toon 1989 default)
  // top_emission_flag = 1: full Planck at TOA (infinite isothermal slab above)
  // top_emission_falg < 0: auto-compute from first layer optical depth
  // (FMS-style)
  //   tau_top = dtau[0] * exp(-1), Btop = (1-exp(-tau_top/ubari)) * BE(0)
  T Btop;
  if (top_emission_flag < 0) {
    T tautop = dtau[0] * btop_factor;
    Btop = (1.0 - exp(-tautop / ubari)) * BE_IN(0);
  } else {
    Btop = top_emission_flag * BE_IN(0);
  }
  // Planck function of the surface: of bc "btemp" when given, else of the
  // bottom level (the caller passes it either way)
  T Bsurf = bsurf_in;

  // Bottom boundary condition depends on surface type:
  // Gas giant (hard_surface=false): Planck + gradient extrapolation below
  //   allows internal heat flux to emerge (matches PICASO hard_surface=0)
  // Terrestrial (hard_surface=true): emissivity * Planck
  //   surface emits as blackbody reduced by emissivity = 1 - albedo
  T bsurf_flux;
  if (hard_surface) {
    bsurf_flux = (1.0 - a_surf_in) * Bsurf;
  } else {
    bsurf_flux = Bsurf + B1[nlay - 1] * ubari;
  }

  // --- Matrix Construction (1-based indices for solver) ---
  Af[0] = 0.0;
  Bf[0] = gam[0] + 1.0;
  Cf[0] = gam[0] - 1.0;
  Df[0] = Btop - Cmm1[0];

  int n_idx = 0;
  for (int i = 1; i < lm2; i += 2) {
    Af[i] = (E1[n_idx] + E3[n_idx]) * (gam[n_idx + 1] - 1.0);
    Bf[i] = (E2[n_idx] + E4[n_idx]) * (gam[n_idx + 1] - 1.0);
    Cf[i] = 2.0 * (1.0 - gam[n_idx + 1] * gam[n_idx + 1]);
    Df[i] = (gam[n_idx + 1] - 1.0) * (Cpm1[n_idx + 1] - Cp[n_idx]) +
            (1.0 - gam[n_idx + 1]) * (Cm[n_idx] - Cmm1[n_idx + 1]);
    n_idx++;
  }

  n_idx = 0;
  for (int i = 2; i < lm1; i += 2) {
    Af[i] = 2.0 * (1.0 - gam[n_idx] * gam[n_idx]);
    Bf[i] = (E1[n_idx] - E3[n_idx]) * (1.0 + gam[n_idx + 1]);
    Cf[i] = (E1[n_idx] + E3[n_idx]) * (gam[n_idx + 1] - 1.0);
    Df[i] = E3[n_idx] * (Cpm1[n_idx + 1] - Cp[n_idx]) +
            E1[n_idx] * (Cm[n_idx] - Cmm1[n_idx + 1]);
    n_idx++;
  }

  // The bottom boundary reflects only off a hard surface. Below a gas giant's
  // column (hard_surface = false) the boundary is the Planck + gradient source
  // above and the angular sweep below, neither of which reflects, so the
  // surface albedo must not enter this row either.
  T const a_row = hard_surface ? a_surf_in : static_cast<T>(0);
  Af[l - 1] = E1[nlay - 1] - a_row * E3[nlay - 1];
  Bf[l - 1] = E2[nlay - 1] - a_row * E4[nlay - 1];
  Cf[l - 1] = 0.0;
  Df[l - 1] = bsurf_flux - Cp[nlay - 1] + a_row * Cm[nlay - 1];

  dtridgl(l, Af, Bf, Cf, Df, xkk);

  for (int n = 0; n < nlay; n++) {
    xk1[n] = xkk[2 * n] + xkk[2 * n + 1];
    xk2[n] = xkk[2 * n] - xkk[2 * n + 1];
    if (fabs(xk2[n]) < 1e-30 * fabs(xkk[2 * n + 1])) xk2[n] = 0.0;

    if (w0[n] <= 1e-4) {
      g[n] = 0.0;
      h[n] = 0.0;
      xj[n] = 0.0;
      xk[n] = 0.0;
      alpha1[n] = twopi * B0[n];
      alpha2[n] = twopi * B1[n];
      sigma1[n] = alpha1[n];
      sigma2[n] = alpha2[n];
    } else {
      T common_den = 1.0 + alp[n];
      g[n] = twopi * w0[n] * xk1[n] * (1.0 + hg[n] * alp[n]) / common_den;
      h[n] = twopi * w0[n] * xk2[n] * (1.0 - hg[n] * alp[n]) / common_den;
      xj[n] = twopi * w0[n] * xk1[n] * (1.0 - hg[n] * alp[n]) / common_den;
      xk[n] = twopi * w0[n] * xk2[n] * (1.0 + hg[n] * alp[n]) / common_den;
      T term_val = ubari * w0[n] * hg[n] / (1.0 - w0[n] * hg[n]);
      alpha1[n] = twopi * (B0[n] + B1[n] * term_val);
      alpha2[n] = twopi * B1[n];
      sigma1[n] = twopi * (B0[n] - B1[n] * term_val);
      sigma2[n] = alpha2[n];
    }
    em1[n] = 1.0 / exp(fmin(lam[n] * dtau[n], 35.0));
    // Half-layer terms of the midpoint fluxes; they do not depend on the
    // angle, so evaluate them once per layer rather than per angle and sweep.
    T exptrm_h = fmin(0.5 * lam[n] * dtau[n], 35.0);
    Ep_mid[n] = exp(exptrm_h);     // exptrm_positive_mdpt
    em1_mid[n] = 1.0 / Ep_mid[n];  // exptrm_minus_mdpt
  }

  // --- Gaussian Quadrature Mu Loop ---
  // Downward sweep first. A hard surface is Lambertian, same as the
  // tridiagonal bottom row, so the reflected intensity needs the downwelling
  // flux and not I_down(mu).
  //
  // Each sweep carries the intensity of the current level in a scalar and
  // adds every level and midpoint into the accumulators as it passes, in the
  // same angle order as before, so every sum sees the same sequence of terms.
  for (int k = 0; k < nlev; k++) facc[k] = 0.0;
  for (int k = 0; k < nlay; k++) facc_mid[k] = 0.0;

  T surf_dn[5];
  T wsum = 0.0;
  for (int m = 0; m < nmu; m++) {
    T u = uarr[m];
    wsum += wuarr[m];

    // Downward loop
    // Top BC: for auto-compute mode, use angle-dependent tau_top
    T Btop_g;
    if (top_emission_flag < 0) {
      T tautop = dtau[0] * btop_factor;
      Btop_g = (1.0 - exp(-tautop / u)) * BE_IN(0);
    } else {
      Btop_g = top_emission_flag * BE_IN(0);
    }
    T lw_down = twopi * Btop_g;
    facc[0] += lw_down * wuarr[m];
    for (int k = 0; k < nlay; k++) {
      T a = dtau[k] / u;
      T em2 = exp(-a);
      T l_u_p1 = lam[k] * u + 1.0;
      T l_u_m1 = lam[k] * u - 1.0;
      T down_ratio = lam[k] * dtau[k] < 35.0
                         ? longwave_down_resonant_ratio(l_u_m1, a, em2)
                         : (em2 - em1[k]) / l_u_m1;

      T lw_next = lw_down * em2 + (xj[k] / l_u_p1) * (Ep[k] - em2) +
                  xk[k] * down_ratio + sigma1[k] * (1.0 - em2) +
                  sigma2[k] * (u * em2 + dtau[k] - u);

      // downward flux at the MIDPOINT of layer k (half-layer optical depth),
      // same source-function form as the level update but to tau_k + dtau/2.
      // (PICASO get_thermal_1d flux_minus_mdpt; vars: xj=J, xk=K.)
      T em2_mid = exp(-0.5 * dtau[k] / u);
      T down_mid_ratio = 0.5 * lam[k] * dtau[k] < 35.0
                             ? longwave_down_resonant_ratio(
                                   l_u_m1, static_cast<T>(0.5) * dtau[k] / u)
                             : (em2_mid - em1_mid[k]) / l_u_m1;
      T mid_dn = lw_down * em2_mid + (xj[k] / l_u_p1) * (Ep_mid[k] - em2_mid) +
                 xk[k] * down_mid_ratio + sigma1[k] * (1.0 - em2_mid) +
                 sigma2[k] * (u * em2_mid + 0.5 * dtau[k] - u);
      facc_mid[k] += mid_dn * wuarr[m];

      lw_down = lw_next;
      facc[k + 1] += lw_down * wuarr[m];
    }
    surf_dn[m] = lw_down;
  }
  for (int k = 0; k < nlev; k++) FLX_DN(k) = facc[k];
  for (int k = 0; k < nlay; k++) FLX_DN_MID(k) = facc_mid[k];

  T fdn_surf = 0.0;
  for (int m = 0; m < nmu; m++) fdn_surf += surf_dn[m] * wuarr[m];
  // Isotropic. sum(w) = 0.5, so a * F_down / sum(w) = 2 a F_down.
  T lw_up_lambert = twopi * (1.0 - a_surf_in) * Bsurf +
                    (wsum != 0.0 ? a_surf_in * fdn_surf / wsum : 0.0);

  for (int k = 0; k < nlev; k++) facc[k] = 0.0;
  for (int k = 0; k < nlay; k++) facc_mid[k] = 0.0;

  for (int m = 0; m < nmu; m++) {
    T u = uarr[m];

    // Upward loop
    // Gas giant: I_up = 2*pi*(B_surf + B1*u) [Planck + gradient]
    // Terrestrial: isotropic emission plus the reflected downwelling flux
    T lw_up;
    if (hard_surface) {
      lw_up = lw_up_lambert;
    } else {
      lw_up = twopi * (Bsurf + B1[nlay - 1] * u);
    }
    facc[nlev - 1] += lw_up * wuarr[m];
    for (int k = nlay - 1; k >= 0; k--) {
      T em2 = exp(-dtau[k] / u);
      T em3 = em1[k] * em2;
      T l_u_m1 = lam[k] * u - 1.0;
      T l_u_p1 = lam[k] * u + 1.0;

      T up_ratio = lam[k] * dtau[k] < 35.0
                       ? longwave_up_resonant_ratio(l_u_m1, dtau[k] / u)
                       : (Ep[k] * em2 - 1.0) / l_u_m1;
      T lw_next = lw_up * em2 + g[k] * up_ratio +
                  (h[k] / l_u_p1) * (1.0 - em3) + alpha1[k] * (1.0 - em2) +
                  alpha2[k] * (u - (dtau[k] + u) * em2);

      // upward flux at the MIDPOINT of layer k (half-layer optical depth),
      // propagated up from the lower-interface level intensity lw_up.
      // (PICASO get_thermal_1d flux_plus_mdpt; vars: g=G, h=H.)
      T em2_mid = exp(-0.5 * dtau[k] / u);
      T up_mid_ratio = lam[k] * dtau[k] < 35.0
                           ? longwave_midpoint_up_resonant_ratio(
                                 l_u_m1, static_cast<T>(0.5) * dtau[k] / u)
                           : (Ep[k] * em2_mid - Ep_mid[k]) / l_u_m1;
      T mid_up = lw_up * em2_mid + g[k] * up_mid_ratio +
                 (h[k] / l_u_p1) * (em1_mid[k] - em1[k] * em2_mid) +
                 alpha1[k] * (1.0 - em2_mid) +
                 alpha2[k] * (u + 0.5 * dtau[k] - (dtau[k] + u) * em2_mid);
      facc_mid[k] += mid_up * wuarr[m];

      lw_up = lw_next;
      facc[k] += lw_up * wuarr[m];
    }
  }
  for (int k = 0; k < nlev; k++) FLX_UP(k) = facc[k];
  for (int k = 0; k < nlay; k++) FLX_UP_MID(k) = facc_mid[k];
}

}  // namespace harp

#undef DTAU_IN
#undef W_IN
#undef G_IN
#undef BE_IN
#undef FLX_UP
#undef FLX_DN
#undef FLX_UP_MID
#undef FLX_DN_MID
