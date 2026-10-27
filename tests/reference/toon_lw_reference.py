"""High-precision reference of pyharp's Toon et al. (1989) longwave solver (toon_mckay89_longwave_impl.h).

The same two-stream solution, tridiagonal system and source-function angular sweep, evaluated with mpmath at
250 significant digits and with exact exponentials: no clip, so a layer of any optical thickness, and exact
resonance lambda mu = 1, use the mathematical limit (e^-a - e^-b)/(b - a) -> e^-a. 250 digits cover the
conditioning of the exact-exponential tridiagonal, about e^(2 lambda dtau), up to lambda dtau ~ 250.

Inputs and output follow the kernel: layers and levels bottom first (level 0 is the bottom), prop rows
(dtau, ssa, g), be the band Planck function at the levels, flux rows (up, down, midpoint up, midpoint down).
"""

import mpmath as mp

mp.mp.dps = 250

UARR = [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
WUARR = [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]


def _diff_ratio(a, b):
    """(e^-a - e^-b) / (b - a), and its limit e^-a at b = a."""
    if a == b:
        return mp.exp(-a)
    return (mp.exp(-a) - mp.exp(-b)) / (b - a)


def toon_lw_reference(prop, be, albedo, hard_surface=True, top_emission_flag=0, btop_factor=mp.exp(-1),
                      bsurf=None, delta_eddington=False):
    nlay = len(prop)
    nlev = nlay + 1
    f = mp.mpf
    twopi = 2 * mp.pi
    ubari = f("0.5")
    # internal order: top first
    dtau_in = [f(prop[nlay - i - 1][0]) for i in range(nlay)]
    w_in = [f(prop[nlay - i - 1][1]) for i in range(nlay)]
    g_in = [f(prop[nlay - i - 1][2]) for i in range(nlay)]
    BE = [f(be[nlev - i - 1]) for i in range(nlev)]
    a_surf = f(albedo)
    Bsurf = f(be[0]) if bsurf is None else f(bsurf)
    uarr = [f(u) for u in UARR]
    wuarr = [f(w) for w in WUARR]

    w0, dtau, hg = [], [], []
    for i in range(nlay):
        if delta_eddington:
            gs = g_in[i] ** 2
            w0.append((1 - gs) * w_in[i] / (1 - w_in[i] * gs))
            dtau.append((1 - w_in[i] * gs) * dtau_in[i])
            hg.append(g_in[i] / (1 + g_in[i]))
        else:
            w0.append(w_in[i]), dtau.append(dtau_in[i]), hg.append(g_in[i])

    alp, lam, gam, term, B0, B1 = [], [], [], [], [], []
    Cpm1, Cmm1, Cp, Cm, Ep, Em, E1, E2, E3, E4 = ([] for _ in range(10))
    for k in range(nlay):
        alp.append(mp.sqrt((1 - w0[k]) / (1 - w0[k] * hg[k])))
        lam.append(alp[k] * (1 - w0[k] * hg[k]) / ubari)
        gam.append((1 - alp[k]) / (1 + alp[k]))
        term.append(ubari / (1 - w0[k] * hg[k]))
        if dtau[k] <= f("1e-6"):
            B1.append(f(0)), B0.append((BE[k + 1] + BE[k]) / 2)
        else:
            B1.append((BE[k + 1] - BE[k]) / dtau[k]), B0.append(BE[k])
        Cpm1.append(B0[k] + B1[k] * term[k])
        Cmm1.append(B0[k] - B1[k] * term[k])
        Cp.append(B0[k] + B1[k] * dtau[k] + B1[k] * term[k])
        Cm.append(B0[k] + B1[k] * dtau[k] - B1[k] * term[k])
        Ep.append(mp.exp(lam[k] * dtau[k]))          # exact: no clip
        Em.append(1 / Ep[k])
        E1.append(Ep[k] + gam[k] * Em[k])
        E2.append(Ep[k] - gam[k] * Em[k])
        E3.append(gam[k] * Ep[k] + Em[k])
        E4.append(gam[k] * Ep[k] - Em[k])

    if top_emission_flag < 0:
        tautop = dtau[0] * f(btop_factor)
        Btop = (1 - mp.exp(-tautop / ubari)) * BE[0]
    else:
        Btop = top_emission_flag * BE[0]
    bsurf_flux = (1 - a_surf) * Bsurf if hard_surface else Bsurf + B1[nlay - 1] * ubari

    n = 2 * nlay
    A, B, C, D = [f(0)] * n, [f(0)] * n, [f(0)] * n, [f(0)] * n
    B[0], C[0], D[0] = gam[0] + 1, gam[0] - 1, Btop - Cmm1[0]
    j = 0
    for i in range(1, n - 2, 2):
        A[i] = (E1[j] + E3[j]) * (gam[j + 1] - 1)
        B[i] = (E2[j] + E4[j]) * (gam[j + 1] - 1)
        C[i] = 2 * (1 - gam[j + 1] ** 2)
        D[i] = (gam[j + 1] - 1) * (Cpm1[j + 1] - Cp[j]) + (1 - gam[j + 1]) * (Cm[j] - Cmm1[j + 1])
        j += 1
    j = 0
    for i in range(2, n - 1, 2):
        A[i] = 2 * (1 - gam[j] ** 2)
        B[i] = (E1[j] - E3[j]) * (1 + gam[j + 1])
        C[i] = (E1[j] + E3[j]) * (gam[j + 1] - 1)
        D[i] = E3[j] * (Cpm1[j + 1] - Cp[j]) + E1[j] * (Cm[j] - Cmm1[j + 1])
        j += 1
    a_row = a_surf if hard_surface else f(0)
    A[n - 1] = E1[nlay - 1] - a_row * E3[nlay - 1]
    B[n - 1] = E2[nlay - 1] - a_row * E4[nlay - 1]
    C[n - 1] = f(0)
    D[n - 1] = bsurf_flux - Cp[nlay - 1] + a_row * Cm[nlay - 1]
    # Thomas algorithm
    C, D = list(C), list(D)
    C[0], D[0] = C[0] / B[0], D[0] / B[0]
    for i in range(1, n):
        den = B[i] - A[i] * C[i - 1]
        C[i] = C[i] / den if i < n - 1 else f(0)
        D[i] = (D[i] - A[i] * D[i - 1]) / den
    x = [f(0)] * n
    x[n - 1] = D[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = D[i] - C[i] * x[i + 1]

    g, h, xj, xk, a1, a2, s1, s2 = ([] for _ in range(8))
    for k in range(nlay):
        xk1, xk2 = x[2 * k] + x[2 * k + 1], x[2 * k] - x[2 * k + 1]
        if w0[k] <= f("1e-4"):
            g.append(f(0)), h.append(f(0)), xj.append(f(0)), xk.append(f(0))
            a1.append(twopi * B0[k]), a2.append(twopi * B1[k]), s1.append(twopi * B0[k]), s2.append(twopi * B1[k])
        else:
            cd = 1 + alp[k]
            g.append(twopi * w0[k] * xk1 * (1 + hg[k] * alp[k]) / cd)
            h.append(twopi * w0[k] * xk2 * (1 - hg[k] * alp[k]) / cd)
            xj.append(twopi * w0[k] * xk1 * (1 - hg[k] * alp[k]) / cd)
            xk.append(twopi * w0[k] * xk2 * (1 + hg[k] * alp[k]) / cd)
            tv = ubari * w0[k] * hg[k] / (1 - w0[k] * hg[k])
            a1.append(twopi * (B0[k] + B1[k] * tv)), a2.append(twopi * B1[k])
            s1.append(twopi * (B0[k] - B1[k] * tv)), s2.append(twopi * B1[k])

    dn, dn_mid = [f(0)] * nlev, [f(0)] * nlay
    surf_dn, wsum = [], f(0)
    for m in range(5):
        u = uarr[m]
        wsum += wuarr[m]
        if top_emission_flag < 0:
            Btop_g = (1 - mp.exp(-dtau[0] * f(btop_factor) / u)) * BE[0]
        else:
            Btop_g = top_emission_flag * BE[0]
        I = twopi * Btop_g
        dn[0] += I * wuarr[m]
        for k in range(nlay):
            a, b = dtau[k] / u, lam[k] * dtau[k]
            e2, e2m = mp.exp(-a), mp.exp(-a / 2)
            nxt = (I * e2 + xj[k] / (lam[k] * u + 1) * (Ep[k] - e2) + xk[k] * a * _diff_ratio(a, b)
                   + s1[k] * (1 - e2) + s2[k] * (u * e2 + dtau[k] - u))
            mid = (I * e2m + xj[k] / (lam[k] * u + 1) * (mp.exp(b / 2) - e2m) + xk[k] * (a / 2) * _diff_ratio(a / 2, b / 2)
                   + s1[k] * (1 - e2m) + s2[k] * (u * e2m + dtau[k] / 2 - u))
            dn_mid[k] += mid * wuarr[m]
            I = nxt
            dn[k + 1] += I * wuarr[m]
        surf_dn.append(I)
    fdn_surf = sum(s * w for s, w in zip(surf_dn, wuarr))
    lam_up = twopi * (1 - a_surf) * Bsurf + a_surf * fdn_surf / wsum

    up, up_mid = [f(0)] * nlev, [f(0)] * nlay
    for m in range(5):
        u = uarr[m]
        I = lam_up if hard_surface else twopi * (Bsurf + B1[nlay - 1] * u)
        up[nlev - 1] += I * wuarr[m]
        for k in range(nlay - 1, -1, -1):
            a, b = dtau[k] / u, lam[k] * dtau[k]
            e2, e2m = mp.exp(-a), mp.exp(-a / 2)
            # g (Ep e^-a - 1)/(lam u - 1) = g Ep a (e^-a - e^-b)/(b - a)
            nxt = (I * e2 + g[k] * Ep[k] * a * _diff_ratio(a, b) + h[k] / (lam[k] * u + 1) * (1 - Em[k] * e2)
                   + a1[k] * (1 - e2) + a2[k] * (u - (dtau[k] + u) * e2))
            mid = (I * e2m + g[k] * Ep[k] * (a / 2) * _diff_ratio(a / 2, b / 2)
                   + h[k] / (lam[k] * u + 1) * (mp.exp(-b / 2) - Em[k] * e2m)
                   + a1[k] * (1 - e2m) + a2[k] * (u + dtau[k] / 2 - (dtau[k] + u) * e2m))
            up_mid[k] += mid * wuarr[m]
            I = nxt
            up[k] += I * wuarr[m]

    out = [[f(0)] * 4 for _ in range(nlev)]
    for k in range(nlev):
        out[nlev - k - 1][0], out[nlev - k - 1][1] = up[k], dn[k]
    for k in range(nlay):
        out[nlev - k - 1][2], out[nlev - k - 1][3] = up_mid[k], dn_mid[k]
    return out
