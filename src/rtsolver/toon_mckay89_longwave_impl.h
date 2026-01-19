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

#define DTAU_IN(i) prop[(nlay - (i) - 1) * 3]
#define W_IN(i) prop[(nlay - (i) - 1) * 3 + 1]
#define G_IN(i) prop[(nlay - (i) - 1) * 3 + 2]
#define FLX_UP(i) flx[2 * (nlev - (i) - 1)]
#define FLX_DN(i) flx[2 * (nlev - (i) - 1) + 1]

namespace harp {

template <typename T>
DISPATCH_MACRO void toon_mckay89_longwave(int nlay, const T *be, const T *prop,
                                          T a_surf_in, T *flx, char *work) {
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
  T *dtau = alloc_from<T>(work, nlay);
  T *tau = alloc_from<T>(work, nlev);
  T *w0 = alloc_from<T>(work, nlay);
  T *hg = alloc_from<T>(work, nlay);
  T *B0 = alloc_from<T>(work, nlay);
  T *B1 = alloc_from<T>(work, nlay);
  T *lam = alloc_from<T>(work, nlay);
  T *gam = alloc_from<T>(work, nlay);
  T *alp = alloc_from<T>(work, nlay);
  T *term = alloc_from<T>(work, nlay);

  T *Cpm1 = alloc_from<T>(work, nlay);
  T *Cmm1 = alloc_from<T>(work, nlay);
  T *Cp = alloc_from<T>(work, nlay);
  T *Cm = alloc_from<T>(work, nlay);

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
  T *xkk = alloc_from<T>(work, l);
  T *xk1 = alloc_from<T>(work, nlay);
  T *xk2 = alloc_from<T>(work, nlay);

  T *g = alloc_from<T>(work, nlay);
  T *h = alloc_from<T>(work, nlay);
  T *xj = alloc_from<T>(work, nlay);
  T *xk = alloc_from<T>(work, nlay);
  T *alpha1 = alloc_from<T>(work, nlay);
  T *alpha2 = alloc_from<T>(work, nlay);
  T *sigma1 = alloc_from<T>(work, nlay);
  T *sigma2 = alloc_from<T>(work, nlay);

  T *em1 = alloc_from<T>(work, nlay);
  T *em2 = alloc_from<T>(work, nlay);
  T *em3 = alloc_from<T>(work, nlay);
  T *lw_up_g = alloc_from<T>(work, nlev);
  T *lw_down_g = alloc_from<T>(work, nlev);

  // Delta-Eddington Scaling
  for (int i = 0; i < nlay; i++) {
    T gsq = G_IN(i) * G_IN(i);
    w0[i] = (1.0 - gsq) * W_IN(i) / (1.0 - W_IN(i) * gsq);
    dtau[i] = (1.0 - W_IN(i) * gsq) * DTAU_IN(i);
    hg[i] = G_IN(i) / (1.0 + G_IN(i));
  }

  tau[0] = 0.0;
  for (int k = 0; k < nlay; k++) {
    tau[k + 1] = tau[k] + dtau[k];
  }

  for (int k = 0; k < nlay; k++) {
    alp[k] = sqrt((1.0 - w0[k]) / (1.0 - w0[k] * hg[k]));
    lam[k] = alp[k] * (1.0 - w0[k] * hg[k]) / ubari;
    gam[k] = (1.0 - alp[k]) / (1.0 + alp[k]);
    term[k] = ubari / (1.0 - w0[k] * hg[k]);

    if (dtau[k] <= 1.0e-6) {
      B1[k] = 0.0;
      B0[k] = 0.5 * (be[k + 1] + be[k]);
    } else {
      B1[k] = (be[k + 1] - be[k]) / dtau[k];
      B0[k] = be[k];
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

  T tautop = dtau[0] * exp(-1.0);
  T Btop = (1.0 - exp(-tautop / ubari)) * be[0];
  T Bsurf = be[nlev - 1];
  T bsurf_flux = Bsurf;  // Bsurf is local variable

  // --- Matrix Construction (1-based indices for solver) ---
  Af[0] = 0.0;
  Bf[0] = gam[0] + 1.0;
  Cf[0] = gam[0] - 1.0;
  Df[0] = Btop - Cmm1[0];

  int n_idx = 0;
  for (int i = 2; i <= lm2; i += 2) {
    Af[i - 1] = (E1[n_idx] + E3[n_idx]) * (gam[n_idx + 1] - 1.0);
    Bf[i - 1] = (E2[n_idx] + E4[n_idx]) * (gam[n_idx + 1] - 1.0);
    Cf[i - 1] = 2.0 * (1.0 - gam[n_idx + 1] * gam[n_idx + 1]);
    Df[i - 1] = (gam[n_idx + 1] - 1.0) * (Cpm1[n_idx + 1] - Cp[n_idx]) +
                (1.0 - gam[n_idx + 1]) * (Cm[n_idx] - Cmm1[n_idx + 1]);
    n_idx++;
  }

  n_idx = 0;
  for (int i = 3; i <= lm1; i += 2) {
    Af[i - 1] = 2.0 * (1.0 - gam[n_idx] * gam[n_idx]);
    Bf[i - 1] = (E1[n_idx] - E3[n_idx]) * (1.0 + gam[n_idx + 1]);
    Cf[i - 1] = (E1[n_idx] + E3[n_idx]) * (gam[n_idx + 1] - 1.0);
    Df[i - 1] = E3[n_idx] * (Cpm1[n_idx + 1] - Cp[n_idx]) +
                E1[n_idx] * (Cm[n_idx] - Cmm1[n_idx + 1]);
    n_idx++;
  }

  Af[l - 1] = E1[nlay - 1] - a_surf_in * E3[nlay - 1];
  Bf[l - 1] = E2[nlay - 1] - a_surf_in * E4[nlay - 1];
  Cf[l - 1] = 0.0;
  Df[l - 1] = bsurf_flux - Cp[nlay - 1] + a_surf_in * Cm[nlay - 1];

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
  }

  for (int k = 0; k < nlev; k++) {
    FLX_UP(k) = 0.0;
    FLX_DN(k) = 0.0;
  }

  // --- Gaussian Quadrature Mu Loop ---
  for (int m = 0; m < nmu; m++) {
    T u = uarr[m];

    // Downward loop
    lw_down_g[0] = twopi * (1.0 - exp(-tautop / u)) * be[0];
    for (int k = 0; k < nlay; k++) {
      em2[k] = exp(-dtau[k] / u);
      T l_u_p1 = lam[k] * u + 1.0;
      T l_u_m1 = lam[k] * u - 1.0;

      lw_down_g[k + 1] =
          lw_down_g[k] * em2[k] + (xj[k] / l_u_p1) * (Ep[k] - em2[k]) +
          (xk[k] / l_u_m1) * (em2[k] - em1[k]) + sigma1[k] * (1.0 - em2[k]) +
          sigma2[k] * (u * em2[k] + dtau[k] - u);
    }

    // Upward loop
    lw_up_g[nlev - 1] = twopi * (Bsurf + B1[nlay - 1] * u);
    for (int k = nlay - 1; k >= 0; k--) {
      em2[k] = exp(-dtau[k] / u);
      T em3_val = em1[k] * em2[k];
      T l_u_m1 = lam[k] * u - 1.0;
      T l_u_p1 = lam[k] * u + 1.0;

      lw_up_g[k] =
          lw_up_g[k + 1] * em2[k] + (g[k] / l_u_m1) * (Ep[k] * em2[k] - 1.0) +
          (h[k] / l_u_p1) * (1.0 - em3_val) + alpha1[k] * (1.0 - em2[k]) +
          alpha2[k] * (u - (dtau[k] + u) * em2[k]);
    }

    for (int k = 0; k < nlev; k++) {
      FLX_DN(k) += lw_down_g[k] * wuarr[m];
      FLX_UP(k) += lw_up_g[k] * wuarr[m];
    }
  }
}

}  // namespace harp

#undef DTAU_IN
#undef W_IN
#undef G_IN
#undef FLX_UP
#undef FLX_DN
