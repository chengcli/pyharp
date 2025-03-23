// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

template <typename T>
void toon_mckay89_longwave(int nlay, int nlev, const T *be, const T *tau_in,
                           const T *w_in, const T *g_in, T a_surf_in, T *flx_up,
                           T *flx_down, char *mem, int memsize) {
  int l = 2 * nlay;
  int lm1 = l - 1;
  int lm2 = l - 2;

  // Constants
  const int nmu = 5;
  const T twopi = 2.0 * M_PI;
  const T ubari = 0.5;

  const T *uarr = {0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821,
                   0.9601901429};
  const T *wuarr = {0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381,
                    0.0967815902};

  // Scratch arrays
  T *dtau_in = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *dtau = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *tau = (T *)get_mem(nlev, sizeof(T), mem, &offset);
  T *w0 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *hg = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *B0 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *B1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *lam = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *gam = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *alp = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *term = (T *)get_mem(nlay, sizeof(T), mem, &offset);

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
  T *xkk = (T *)get_mem(l, sizeof(T), mem, &offset);
  T *xk1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *xk2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);

  T *g = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *h = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *xj = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *xk = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *alpha1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *alpha2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *sigma1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *sigma2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);

  T *em1 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *obj = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *epp = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *obj2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *epp2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *em2 = (T *)get_mem(nlay, sizeof(T), mem, &offset);
  T *em3 = (T *)get_mem(nlay, sizeof(T), mem, &offset);

  T *lw_up_g = (T *)get_mem(nlev, sizeof(T), mem, &offset);
  T *lw_down_g = (T *)get_mem(nlev, sizeof(T), mem, &offset);

  // === Precomputations ===

  for (int k = 0; k < nlay; ++k) dtau_in[k] = tau_in[k + 1] - tau_in[k];

  for (int k = 0; k < nlay; ++k) {
    T g2 = g_in[k] * g_in[k];
    T denom = 1.0 - w_in[k] * g2;
    w0[k] = (1.0 - g2) * w_in[k] / denom;
    dtau[k] = denom * dtau_in[k];
    hg[k] = g_in[k] / (1.0 + g_in[k]);
  }

  tau[0] = 0.0;
  for (int k = 0; k < nlay; ++k) tau[k + 1] = tau[k] + dtau[k];

  for (int k = 0; k < nlay; ++k) {
    alp[k] = sqrt((1.0 - w0[k]) / (1.0 - w0[k] * hg[k]));
    lam[k] = alp[k] * (1.0 - w0[k] * hg[k]) / ubari;
    gam[k] = (1.0 - alp[k]) / (1.0 + alp[k]);
    term[k] = ubari / (1.0 - w0[k] * hg[k]);

    if (dtau[k] <= 1e-6) {
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
  }

  T tautop = dtau[0] * exp(-1.0);
  T Btop = (1.0 - exp(-tautop / ubari)) * be[0];
  T Bsurf = be[nlev - 1];

  T bottom = Bsurf + B1[nlay - 1] * ubari;

  // === Solve tridiagonal system (not shown again for brevity) ===
  dtridgl(l, Af, Bf, Cf, Df, xk);

  // === Calculate xk1, xk2 from xkk ===
  for (int n = 0; n < nlay; ++n) {
    xk1[n] = xkk[2 * n] + xkk[2 * n + 1];
    xk2[n] = xkk[2 * n] - xkk[2 * n + 1];
    if (fabs(xk2[n] / xkk[2 * n]) < 1e-30) xk2[n] = 0.0;
  }

  // === Conditional computation for g, h, xj, xk etc. ===
  for (int k = 0; k < nlay; ++k) {
    if (w0[k] <= 1e-4) {
      g[k] = h[k] = xj[k] = xk[k] = 0.0;
      alpha1[k] = sigma1[k] = twopi * B0[k];
      alpha2[k] = sigma2[k] = twopi * B1[k];
    } else {
      T f1 = (1.0 + hg[k] * alp[k]) / (1.0 + alp[k]);
      T f2 = (1.0 - hg[k] * alp[k]) / (1.0 + alp[k]);

      g[k] = twopi * w0[k] * xk1[k] * f1;
      h[k] = twopi * w0[k] * xk2[k] * f2;
      xj[k] = twopi * w0[k] * xk1[k] * f2;
      xk[k] = twopi * w0[k] * xk2[k] * f1;

      T fact = ubari * w0[k] * hg[k] / (1.0 - w0[k] * hg[k]);
      alpha1[k] = twopi * (B0[k] + B1[k] * fact);
      sigma1[k] = twopi * (B0[k] - B1[k] * fact);
      alpha2[k] = sigma2[k] = twopi * B1[k];
    }
  }

  // === Gaussian quadrature integration ===
  memset(flx_up, 0, nlev * sizeof(T), mem, &offset);
  memset(flx_down, 0, nlev * sizeof(T), mem, &offset);

  for (int m = 0; m < nmu; ++m) {
    for (int k = 0; k < nlay; ++k) {
      em2[k] = exp(-dtau[k] / uarr[m]);
      em3[k] = em1[k] * em2[k];
    }

    lw_down_g[0] = twopi * (1.0 - exp(-tautop / uarr[m])) * be[0];
    for (int k = 0; k < nlay; ++k) {
      lw_down_g[k + 1] =
          lw_down_g[k] * em2[k] +
          (xj[k] / (lam[k] * uarr[m] + 1.0)) * (epp[k] - em2[k]) +
          (xk[k] / (lam[k] * uarr[m] - 1.0)) * (em2[k] - em[k]) +
          sigma1[k] * (1.0 - em2[k]) +
          sigma2[k] * (uarr[m] * em2[k] + dtau[k] - uarr[m]);
    }

    lw_up_g[nlev - 1] = twopi * (Bsurf + B1[nlay - 1] * uarr[m]);
    for (int k = nlay - 1; k >= 0; --k) {
      lw_up_g[k] = lw_up_g[k + 1] * em2[k] +
                   (g[k] / (lam[k] * uarr[m] - 1.0)) * (epp[k] * em2[k] - 1.0) +
                   (h[k] / (lam[k] * uarr[m] + 1.0)) * (1.0 - em3[k]) +
                   alpha1[k] * (1.0 - em2[k]) +
                   alpha2[k] * (uarr[m] - (dtau[k] + uarr[m]) * em2[k]);
    }

    for (int k = 0; k < nlev; ++k) {
      flx_down[k] += lw_down_g[k] * wuarr[m];
      flx_up[k] += lw_up_g[k] * wuarr[m];
    }
  }
}
