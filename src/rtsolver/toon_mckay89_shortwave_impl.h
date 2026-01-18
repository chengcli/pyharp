// C/C++
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// base
#include <configure.h>

// harp
#include <harp/utils/alloc.h>

#include "dtridgl_impl.h"

#define DTAU_IN(i) prop[(nlay - i - 1) * 3]
#define W_IN(i) prop[(nlay - i - 1) * 3 + 1]
#define G_IN(i) prop[(nlay - i - 1) * 3 + 2]
#define FLX_UP(i) flx[2 * (nlev - i - 1)]
#define FLX_DN(i) flx[2 * (nlev - i - 1) + 1]

namespace harp {

template <typename T>
DISPATCH_MACRO void toon_mckay89_shortwave(int nlay, T F0_in, T const *mu_in,
                                           T const *prop, T w_surf_in, T *flx,
                                           char *work) {
  int nlev = nlay + 1;
  int l = 2 * nlay;
  int lm2 = l - 2;
  int lm1 = l - 1;

  // --- Memory Allocation ---
  T *dir = alloc_from<T>(work, nlev);
  T *tau = alloc_from<T>(work, nlev);
  T *cum_trans = alloc_from<T>(work, nlev);
  T *tau_in = alloc_from<T>(work, nlev);
  T *dtau = alloc_from<T>(work, nlay);
  T *mu_zm = alloc_from<T>(work, nlay);
  T *w0 = alloc_from<T>(work, nlay);
  T *hg = alloc_from<T>(work, nlay);
  T *g1 = alloc_from<T>(work, nlay);
  T *g2 = alloc_from<T>(work, nlay);
  T *g3 = alloc_from<T>(work, nlay);
  T *g4 = alloc_from<T>(work, nlay);
  T *lam = alloc_from<T>(work, nlay);
  T *gam = alloc_from<T>(work, nlay);
  T *denom = alloc_from<T>(work, nlay);
  T *Am = alloc_from<T>(work, nlay);
  T *Ap = alloc_from<T>(work, nlay);
  T *Cpm1 = alloc_from<T>(work, nlay);
  T *Cmm1 = alloc_from<T>(work, nlay);
  T *Cp = alloc_from<T>(work, nlay);
  T *Cm = alloc_from<T>(work, nlay);
  T *exptrm = alloc_from<T>(work, nlay);
  T *Ep = alloc_from<T>(work, nlay);
  T *Em = alloc_from<T>(work, nlay);
  T *E1 = alloc_from<T>(work, nlay);
  T *E2 = alloc_from<T>(work, nlay);
  T *E3 = alloc_from<T>(work, nlay);
  T *E4 = alloc_from<T>(work, nlay);
  T *Af = alloc_from<T>(work, l);
  T *Bf = alloc_from<T>(work, l);
  T *Cf = alloc_from<T>(work, l);
  T *Df = alloc_from<T>(work, l);
  T *xk = alloc_from<T>(work, l);
  T *xk1 = alloc_from<T>(work, nlay);
  T *xk2 = alloc_from<T>(work, nlay);

  const T sqrt3 = sqrt(3.0);
  const T sqrt3d2 = sqrt3 / 2.0;
  const T btop = 0.0, bsurf = 0.0;

  // Check for zero albedo
  bool all_zero_w = true;
  for (int i = 0; i < nlay; i++) {
    if (W_IN(i) > 1.0e-12) {
      all_zero_w = false;
      break;
    }
  }

  // compute integrated optical depth
  tau_in[0] = 0.0;
  for (int i = 0; i < nlay; i++) {
    tau_in[i + 1] = tau_in[i] + DTAU_IN(i);
  }

  if (all_zero_w) {
    // --- Special Case: Direct Beam Only ---
    if (mu_in[nlev - 1] == mu_in[0]) {
      for (int k = 0; k < nlev; k++) {
        FLX_DN(k) = F0_in * mu_in[nlev - 1] * exp(-tau_in[k] / mu_in[nlev - 1]);
      }
    } else {
      cum_trans[0] = tau_in[0] / mu_in[0];
      for (int k = 0; k < nlev - 1; k++) {
        cum_trans[k + 1] = cum_trans[k] + DTAU_IN(k) / mu_in[k + 1];
      }
      for (int k = 0; k < nlev; k++) {
        FLX_DN(k) = F0_in * mu_in[nlev - 1] * exp(-cum_trans[k]);
      }
    }
    FLX_DN(nlev - 1) *= (1.0 - w_surf_in);
    for (int k = 0; k < nlev; k++) FLX_UP(k) = 0.0;

  } else {
    // --- General Case: Toon et al. 1989 Solver ---

    for (int i = 0; i < nlay; i++) {
      T g_sq = G_IN(i) * G_IN(i);
      w0[i] = ((1.0 - g_sq) * W_IN(i)) / (1.0 - W_IN(i) * g_sq);
      dtau[i] = (1.0 - W_IN(i) * g_sq) * DTAU_IN(i);
      hg[i] = G_IN(i) / (1.0 + G_IN(i));
    }

    tau[0] = 0.0;
    for (int k = 0; k < nlay; k++) tau[k + 1] = tau[k] + dtau[k];

    if (mu_in[nlev - 1] == mu_in[0]) {
      T mu_val = mu_in[nlev - 1];
      for (int k = 0; k < nlev; k++)
        dir[k] = F0_in * mu_val * exp(-tau[k] / mu_val);
      for (int i = 0; i < nlay; i++) mu_zm[i] = mu_val;
    } else {
      cum_trans[0] = tau[0] / mu_in[0];
      for (int k = 0; k < nlev - 1; k++)
        cum_trans[k + 1] = cum_trans[k] + (tau[k + 1] - tau[k]) / mu_in[k + 1];
      for (int k = 0; k < nlev; k++)
        dir[k] = F0_in * mu_in[nlev - 1] * exp(-cum_trans[k]);
      for (int i = 0; i < nlay; i++) mu_zm[i] = (mu_in[i] + mu_in[i + 1]) / 2.0;
    }

    for (int i = 0; i < nlay; i++) {
      g1[i] = sqrt3d2 * (2.0 - w0[i] * (1.0 + hg[i]));
      g2[i] = (sqrt3d2 * w0[i]) * (1.0 - hg[i]);
      if (g2[i] == 0.0) g2[i] = 1.0e-10;
      g3[i] = (1.0 - sqrt3 * hg[i] * mu_zm[i]) / 2.0;
      g4[i] = 1.0 - g3[i];
      lam[i] = sqrt(g1[i] * g1[i] - g2[i] * g2[i]);
      gam[i] = (g1[i] - lam[i]) / g2[i];
      denom[i] = (lam[i] * lam[i]) - 1.0 / (mu_zm[i] * mu_zm[i]);
      if (denom[i] == 0.0) denom[i] = 1.0e-10;
      Ap[i] = F0_in * w0[i] *
              (g3[i] * (g1[i] - 1.0 / mu_zm[i]) + g2[i] * g4[i]) / denom[i];
      Am[i] = F0_in * w0[i] *
              (g4[i] * (g1[i] + 1.0 / mu_zm[i]) + g2[i] * g3[i]) / denom[i];
      Cpm1[i] = Ap[i] * exp(-tau[i] / mu_zm[i]);
      Cmm1[i] = Am[i] * exp(-tau[i] / mu_zm[i]);
      Cp[i] = Ap[i] * exp(-tau[i + 1] / mu_zm[i]);
      Cm[i] = Am[i] * exp(-tau[i + 1] / mu_zm[i]);
      exptrm[i] = fmin(lam[i] * dtau[i], 35.0);
      Ep[i] = exp(exptrm[i]);
      Em[i] = 1.0 / Ep[i];
      E1[i] = Ep[i] + gam[i] * Em[i];
      E2[i] = Ep[i] - gam[i] * Em[i];
      E3[i] = gam[i] * Ep[i] + Em[i];
      E4[i] = gam[i] * Ep[i] - Em[i];
    }

    // Matrix Setup
    Af[1] = 0.0;
    Bf[1] = gam[0] + 1.0;
    Cf[1] = gam[0] - 1.0;
    Df[1] = btop - Cmm1[0];
    int n_idx = 0;
    for (int i = 2; i <= lm2; i += 2) {
      Af[i] = (E1[n_idx] + E3[n_idx]) * (gam[n_idx + 1] - 1.0);
      Bf[i] = (E2[n_idx] + E4[n_idx]) * (gam[n_idx + 1] - 1.0);
      Cf[i] = 2.0 * (1.0 - gam[n_idx + 1] * gam[n_idx + 1]);
      Df[i] = (gam[n_idx + 1] - 1.0) * (Cpm1[n_idx + 1] - Cp[n_idx]) +
              (1.0 - gam[n_idx + 1]) * (Cm[n_idx] - Cmm1[n_idx + 1]);
      n_idx++;
    }
    n_idx = 0;
    for (int i = 3; i <= lm1; i += 2) {
      Af[i] = 2.0 * (1.0 - gam[n_idx] * gam[n_idx]);
      Bf[i] = (E1[n_idx] - E3[n_idx]) * (1.0 + gam[n_idx + 1]);
      Cf[i] = (E1[n_idx] + E3[n_idx]) * (gam[n_idx + 1] - 1.0);
      Df[i] = E3[n_idx] * (Cpm1[n_idx + 1] - Cp[n_idx]) +
              E1[n_idx] * (Cm[n_idx] - Cmm1[n_idx + 1]);
      n_idx++;
    }
    Af[l] = E1[nlay - 1] - w_surf_in * E3[nlay - 1];
    Bf[l] = E2[nlay - 1] - w_surf_in * E4[nlay - 1];
    Cf[l] = 0.0;
    Df[l] = bsurf - Cp[nlay - 1] + w_surf_in * Cm[nlay - 1];

    dtridgl(l, Af, Bf, Cf, Df, xk);

    for (int n = 0; n < nlay; n++) {
      xk1[n] = xk[2 * n + 1] + xk[2 * n + 2];
      xk2[n] = xk[2 * n + 1] - xk[2 * n + 2];
      if (fabs(xk2[n]) < 1e-30 * fabs(xk[2 * n + 1])) xk2[n] = 0.0;
      FLX_UP(n) = xk1[n] + gam[n] * xk2[n] + Cpm1[n];
      FLX_DN(n) = xk1[n] * gam[n] + xk2[n] + Cmm1[n];
    }
    FLX_UP(nlev - 1) = xk1[nlay - 1] * Ep[nlay - 1] +
                       gam[nlay - 1] * xk2[nlay - 1] * Em[nlay - 1] +
                       Cp[nlay - 1];
    FLX_DN(nlev - 1) = xk1[nlay - 1] * Ep[nlay - 1] * gam[nlay - 1] +
                       xk2[nlay - 1] * Em[nlay - 1] + Cm[nlay - 1];
    for (int k = 0; k < nlev; k++) FLX_DN(k) += dir[k];
  }
}

}  // namespace harp

#undef DTAU_IN
#undef W_IN
#undef G_IN
#undef FLX_UP
#undef FLX_DN
