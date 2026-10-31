"""Exact Lorenz-Mie Q_ext, Q_sca of a homogeneous sphere at 60 significant digits (mpmath): prints the table of tests/test_mie_small_particle.cpp."""
import mpmath as mp

mp.mp.dps = 60


def psi(n, z):  # Riccati-Bessel psi_n(z) = z j_n(z)
    return z * mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z)


def chi(n, x):  # chi_n(x) = -x y_n(x)
    return -x * mp.sqrt(mp.pi / (2 * x)) * mp.bessely(n + mp.mpf(1) / 2, x)


def dpsi(n, z):
    return psi(n - 1, z) - n * psi(n, z) / z


def mie_q(nreal, kimag, x, nmax=12):
    m = mp.mpc(nreal, kimag)          # n + i k (Bohren and Huffman); Q_ext and Q_sca do not depend on the sign
    x = mp.mpf(x)
    mx = m * x
    qext = qsca = mp.mpf(0)
    for n in range(1, nmax + 1):
        xi = psi(n, x) - 1j * chi(n, x)
        dxi = dpsi(n, x) - 1j * (chi(n - 1, x) - n * chi(n, x) / x)
        a = (m * psi(n, mx) * dpsi(n, x) - psi(n, x) * dpsi(n, mx)) / (m * psi(n, mx) * dxi - xi * dpsi(n, mx))
        b = (psi(n, mx) * dpsi(n, x) - m * psi(n, x) * dpsi(n, mx)) / (psi(n, mx) * dxi - m * xi * dpsi(n, mx))
        qext += (2 * n + 1) * mp.re(a + b)
        qsca += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
    return 2 * qext / x ** 2, 2 * qsca / x ** 2


CASES = [(n, k, x) for (n, k) in [(1.5, 0.0), (1.5, 1e-8), (1.5, 1e-6), (1.33, 1e-4), (1.33, 1e-2), (1.5, 1.0)]
         for x in [5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 1e-2, 5e-2, 0.2, 1.0]]

if __name__ == "__main__":
    for n, k, x in CASES:
        qe, qs = mie_q(n, k, x)
        print(f"    {{{n!r}, {k!r}, {x!r}, {mp.nstr(qe, 20)}, {mp.nstr(qs, 20)}}},")
