// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

template <typename T>
void toon_mckay89_shortwave(int nlay, int nlev, T F0_in, const T *mu_in,
                            const T *tau_in, const T *w_in, const T *g_in,
                            T w_surf_in, T *flx_down, T *flx_up, char *mem,
                            int memsize) {
  int l = 2 * nlay;
  int lm1 = l - 1;
  int lm2 = l - 2;

  // Constants
  const T sqrt3 = sqrt(3.0);
  const T sqrt3d2 = sqrt3 / 2.0;
  const T bsurf = 0.0, btop = 0.0;

  // Scratch arrays
  T *dir = (T *)get_mem(nlev, sizeof(T), mem, &offset);
  T *tau = (T *)get_mem(nlev, sizeof(T), mem, &offset);
  T *cum_trans = (T *)get_mem(nlev, sizeof(T), mem, &offset);
  T *dtau_in = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *dtau = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *mu_zm = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *w0 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *hg = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *g1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *g2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *g3 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *g4 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *lam = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *gam = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *denom = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Am = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Ap = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Cpm1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Cmm1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Cp = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Cm = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *exptrm = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Ep = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Em = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *E1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *E2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *E3 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *E4 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *Af = (T *)get_mem(l, sizeof(T), mem, &offset);
  T *Bf = (T *)get_mem(l, sizeof(T), mem, &offset);
  T *Cf = (T *)get_mem(l, sizeof(T), mem, &offset);
  T *Df = (T *)get_mem(l, sizeof(T), mem, &offset);
  T *xk = (T *)get_mem(l, sizeof(T), mem, &offset);
  T *xk1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *xk2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *opt1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);

  if (offset > memsize) {
    fprintf(stderr,
            "Error: Memory allocation failed in toon_mckay89_shortwave\n");
    exit(EXIT_FAILURE);
  }

  // Early exit if all single scattering albedos are ~0
  int all_zero = 1;
  for (int k = 0; k < nlay; ++k) {
    if (w_in[k] > 1.0e-12) {
      all_zero = 0;
      break;
    }
  }

  if (all_zero) {
    if (mu_in[nlev - 1] == mu_in[0]) {
      for (int k = 0; k < nlev; ++k)
        flx_down[k] =
            F0_in * mu_in[nlev - 1] * exp(-tau_in[k] / mu_in[nlev - 1]);
    } else {
      cum_trans[0] = tau_in[0] / mu_in[0];
      for (int k = 1; k < nlev; ++k)
        cum_trans[k] =
            cum_trans[k - 1] + (tau_in[k] - tau_in[k - 1]) / mu_in[k];
      for (int k = 0; k < nlev; ++k)
        flx_down[k] = F0_in * mu_in[nlev - 1] * exp(-cum_trans[k]);
    }
    flx_down[nlev - 1] *= (1.0 - w_surf_in);
    for (int k = 0; k < nlev; ++k) flx_up[k] = 0.0;
  }

  // Continue with rest of code
  for (int k = 0; k < nlay; ++k) dtau_in[k] = tau_in[k + 1] - tau_in[k];

  for (int k = 0; k < nlay; ++k) {
    T g2_val = g_in[k] * g_in[k];
    T denom_val = 1.0 - w_in[k] * g2_val;
    w0[k] = ((1.0 - g2_val) * w_in[k]) / denom_val;
    dtau[k] = denom_val * dtau_in[k];
    hg[k] = g_in[k] / (1.0 + g_in[k]);
  }

  tau[0] = 0.0;
  for (int k = 0; k < nlay; ++k) tau[k + 1] = tau[k] + dtau[k];

  if (mu_in[nlev - 1] == mu_in[0]) {
    for (int k = 0; k < nlev; ++k)
      dir[k] = F0_in * mu_in[nlev - 1] * exp(-tau[k] / mu_in[nlev - 1]);
    for (int k = 0; k < nlay; ++k) mu_zm[k] = mu_in[nlev - 1];
  } else {
    cum_trans[0] = tau[0] / mu_in[0];
    for (int k = 1; k < nlev; ++k)
      cum_trans[k] = cum_trans[k - 1] + (tau[k] - tau[k - 1]) / mu_in[k];
    for (int k = 0; k < nlev; ++k)
      dir[k] = F0_in * mu_in[nlev - 1] * exp(-cum_trans[k]);
    for (int k = 0; k < nlay; ++k) mu_zm[k] = 0.5 * (mu_in[k] + mu_in[k + 1]);
  }

  for (int k = 0; k < nlay; ++k) {
    g1[k] = sqrt3d2 * (2.0 - w0[k] * (1.0 + hg[k]));
    g2[k] = sqrt3d2 * w0[k] * (1.0 - hg[k]);
    if (g2[k] == 0.0) g2[k] = 1e-10;
    g3[k] = 0.5 * (1.0 - sqrt3 * hg[k] / mu_zm[k]);
    g4[k] = 1.0 - g3[k];

    lam[k] = sqrt(g1[k] * g1[k] - g2[k] * g2[k]);
    gam[k] = (g1[k] - lam[k]) / g2[k];

    denom[k] = lam[k] * lam[k] - 1.0 / (mu_zm[k] * mu_zm[k]);
    if (denom[k] == 0.0) denom[k] = 1e-10;

    Am[k] = F0_in * w0[k] * (g4[k] * (g1[k] + 1.0 / mu_zm[k]) + g2[k] * g3[k]) /
            denom[k];
    Ap[k] = F0_in * w0[k] * (g3[k] * (g1[k] - 1.0 / mu_zm[k]) + g2[k] * g4[k]) /
            denom[k];
  }

  for (int k = 0; k < nlay; ++k) {
    opt1[k] = exp(-tau[k] / mu_zm[k]);
    Cpm1[k] = Ap[k] * opt1[k];
    Cmm1[k] = Am[k] * opt1[k];

    opt1[k] = exp(-tau[k + 1] / mu_zm[k]);
    Cp[k] = Ap[k] * opt1[k];
    Cm[k] = Am[k] * opt1[k];

    exptrm[k] = fmin(lam[k] * dtau[k], 35.0);
    Ep[k] = exp(exptrm[k]);
    Em[k] = 1.0 / Ep[k];

    E1[k] = Ep[k] + gam[k] * Em[k];
    E2[k] = Ep[k] - gam[k] * Em[k];
    E3[k] = gam[k] * Ep[k] + Em[k];
    E4[k] = gam[k] * Ep[k] - Em[k];
  }

  // System assembly
  Af[0] = 0.0;
  Bf[0] = gam[0] + 1.0;
  Cf[0] = gam[0] - 1.0;
  Df[0] = btop - Cmm1[0];

  n = 0;
  for (int i = 1; i < lm2; i += 2) {
    Af[i] = (E1[n] + E3[n]) * (gam[n + 1] - 1.0);
    Bf[i] = (E2[n] + E4[n]) * (gam[n + 1] - 1.0);
    Cf[i] = 2.0 * (1.0 - gam[n + 1] * gam[n + 1]);
    Df[i] = (gam[n + 1] - 1.0) * (Cpm1[n + 1] - Cp[n]) +
            (1.0 - gam[n + 1]) * (Cm[n] - Cmm1[n + 1]);
    ++n;
  }

  n = 0;
  for (int i = 2; i < lm1; i += 2) {
    Af[i] = 2.0 * (1.0 - gam[n] * gam[n]);
    Bf[i] = (E1[n] - E3[n]) * (1.0 + gam[n + 1]);
    Cf[i] = (E1[n] + E3[n]) * (gam[n + 1] - 1.0);
    Df[i] = E3[n] * (Cpm1[n + 1] - Cp[n]) + E1[n] * (Cm[n] - Cmm1[n + 1]);
    ++n;
  }

  Af[l - 1] = E1[nlay - 1] - w_surf_in * E3[nlay - 1];
  Bf[l - 1] = E2[nlay - 1] - w_surf_in * E4[nlay - 1];
  Cf[l - 1] = 0.0;
  Df[l - 1] = bsurf - Cp[nlay - 1] + w_surf_in * Cm[nlay - 1];

  dtridgl(l, Af, Bf, Cf, Df, xk);

  for (int n = 0; n < nlay; ++n) {
    xk1[n] = xk[2 * n] + xk[2 * n + 1];
    xk2[n] = xk[2 * n] - xk[2 * n + 1];
    if (fabs(xk2[n] / xk[2 * n]) < 1e-30) xk2[n] = 0.0;
  }

  for (int n = 0; n < nlay; ++n) {
    flx_up[n] = xk1[n] + gam[n] * xk2[n] + Cpm1[n];
    flx_down[n] = xk1[n] * gam[n] + xk2[n] + Cmm1[n];
  }

  flx_up[nlev - 1] = xk1[nlay - 1] * Ep[nlay - 1] +
                     gam[nlay - 1] * xk2[nlay - 1] * Em[nlay - 1] +
                     Cp[nlay - 1];
  flx_down[nlev - 1] = xk1[nlay - 1] * Ep[nlay - 1] * gam[nlay - 1] +
                       xk2[nlay - 1] * Em[nlay - 1] + Cm[nlay - 1];

  for (int n = 0; n < nlev; ++n) flx_down[n] += dir[n];
}
