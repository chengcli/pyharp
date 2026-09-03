#pragma once

// C/C++
#include <cmath>
#include <cstdint>

// base
#include <configure.h>

// harp
#include <harp/math/complex.h>

namespace harp {

template <typename T>
struct MieEfficiencyDevice {
  T qext;
  T qsca;
  T g;
  int status;
};

template <typename T>
struct WaterLiquidMieProperties {
  T extinction;
  T single_scattering_albedo;
  T g;
};

template <typename T>
DISPATCH_MACRO T clamp_value(T value, T lo, T hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

template <typename T>
DISPATCH_MACRO Complex<T> lentz_log_derivative_device(Complex<T> z, int n,
                                                      int* status) {
  // Continued-fraction coefficients:
  // a_j = (-1)^(j+1) * (2*n + 2*j - 1) / z
  auto const a1 = static_cast<T>(2 * n + 1) / z;
  auto const a2 = -static_cast<T>(2 * n + 3) / z;
  // U_2 = P_2 / P_1
  // V_2 = Q_2 / Q_1
  // These are the numerator and denominator ratios in Lentz's
  // continued-fraction recurrence.
  auto u = a2 + static_cast<T>(1) / a1;
  auto v = a2;
  // delta_2 = F_2 / F_1
  // This ratio measures the change between successive convergents.
  auto ratio = u / v;
  // F_2 = F_1 * delta_2
  // Later iterations accumulate the same product until it converges.
  auto runratio = a1 * ratio;
  T const tolerance = sizeof(T) == sizeof(float) ? static_cast<T>(1.0e-6)
                                                 : static_cast<T>(1.0e-12);
  int iterations = 0;
  int j = 3;
  while (complex_abs(ratio - static_cast<T>(1)) > tolerance) {
    if (++iterations >= 100000) {
      *status = 1;
      break;
    }
    T const sign = (j % 2 == 1) ? static_cast<T>(1) : static_cast<T>(-1);
    auto const aj = sign * static_cast<T>(2 * n + 2 * j - 1) / z;
    u = aj + static_cast<T>(1) / u;
    v = aj + static_cast<T>(1) / v;
    ratio = u / v;
    runratio = runratio * ratio;
    ++j;
  }
  return -static_cast<T>(n) / z + runratio;
}

template <typename T>
DISPATCH_MACRO int mie_nstop(T x) {
  // Wiscombe (1980) criterion for 8 < x < 4200.
  // It specifies where the otherwise infinite Lorenz-Mie series is truncated.
  return static_cast<int>(
      x + static_cast<T>(4.05) * pow(x, static_cast<T>(1.0 / 3.0)) +
      static_cast<T>(2.0));
}

// Full homogeneous-sphere Lorenz-Mie efficiencies. The recurrence follows
// Bohren and Huffman (1983) and the numerical layout used by miepython:
// m uses the absorbing n-i*k convention and x is the external size parameter.
template <typename T>
DISPATCH_MACRO MieEfficiencyDevice<T> mie_efficiency_device(T nreal, T kimag,
                                                            T x,
                                                            Complex<T>* work,
                                                            int max_order) {
  MieEfficiencyDevice<T> out{0, 0, 0, 0};
  if (!(x > static_cast<T>(0)) || !(nreal > static_cast<T>(0)) ||
      kimag < static_cast<T>(0)) {
    out.status = 2;
    return out;
  }

  Complex<T> const m(nreal, -kimag);
  // Avoid cancellation in the Riccati-Bessel recurrence in the Rayleigh
  // limit. This branch is asymptotically exact as x -> 0.
  if (x < static_cast<T>(1.0e-3)) {
    auto const m2 = m * m;
    auto const alpha = (m2 - static_cast<T>(1)) / (m2 + static_cast<T>(2));
    T const qsca = static_cast<T>(8.0 / 3.0) * pow(x, static_cast<T>(4)) *
                   complex_norm(alpha);
    T const qabs = -static_cast<T>(4) * x * alpha.i > static_cast<T>(0)
                       ? -static_cast<T>(4) * x * alpha.i
                       : static_cast<T>(0);
    out.qext = qsca + qabs;
    out.qsca = qsca;
    out.g = 0;
    return out;
  }

  // The highest retained multipole order grows with the size parameter x.
  int const computed_nstop = mie_nstop(x);
  int const nstop = computed_nstop > 1 ? computed_nstop : 1;
  int const derivative_order = nstop + 1;
  if (derivative_order + 1 > max_order) {
    out.status = 3;
    return out;
  }

  // Divide the caller-provided scratch space into the logarithmic derivatives
  // D_n and the two complex Mie-coefficient sequences a_n and b_n.
  auto* d = work;
  auto* a = d + max_order;
  auto* b = a + max_order;
  auto const z = m * x;
  T const kappa = kimag;
  // D_n(z) = psi'_n(z)/psi_n(z). Wiscombe's criterion chooses the stable
  // recurrence direction; the downward start uses Lentz's continued
  // fraction, matching miepython's validated implementation.
  bool const use_downward =
      nreal < static_cast<T>(1) || nreal > static_cast<T>(10) ||
      kappa > static_cast<T>(10) ||
      x * kappa >= static_cast<T>(3.9) - static_cast<T>(10.8) * nreal +
                       static_cast<T>(13.78) * nreal * nreal;

  if (use_downward) {
    auto last = lentz_log_derivative_device(z, derivative_order, &out.status);
    for (int n = derivative_order; n > 0; --n) {
      auto const nz = static_cast<T>(n) / z;
      last = nz - static_cast<T>(1) / (last + nz);
      d[n - 1] = last;
    }
  } else {
    auto const exponential = complex_exp(Complex<T>(0, -2) * z);
    d[1] = -static_cast<T>(1) / z +
           (static_cast<T>(1) - exponential) /
               ((static_cast<T>(1) - exponential) / z -
                Complex<T>(0, 1) * (static_cast<T>(1) + exponential));
    for (int n = 2; n <= derivative_order; ++n) {
      auto const nz = static_cast<T>(n) / z;
      d[n] = static_cast<T>(1) / (nz - d[n - 1]) - nz;
    }
  }
  if (out.status != 0) return out;

  T const sin_x = sin(x);
  T const cos_x = cos(x);
  T psi_nm1 = sin_x;  // n-1 (n minus 1) Riccati-Bessel function
  T psi_n = psi_nm1 / x - cos_x;
  // xi_n = psi_n + i*chi_n is the outgoing Riccati-Hankel function in the
  // sign convention used by this implementation.
  Complex<T> xi_nm1(psi_nm1, cos_x);  // psi_nm1 + i*chi_nm1
  Complex<T> xi_n(psi_n, cos_x / x + sin_x);

  for (int n = 1; n <= nstop; ++n) {
    auto const dn = d[n];
    auto const nx = static_cast<T>(n) / x;
    auto const da = dn / m + nx;
    auto const db = m * dn + nx;
    // a_n is the electric-type coefficient and b_n is the magnetic-type
    // coefficient in the Lorenz-Mie expansion.
    a[n - 1] = (da * psi_n - psi_nm1) /
               (da * xi_n - xi_nm1);  // Mie-scattering coefficient a_n
    b[n - 1] = (db * psi_n - psi_nm1) /
               (db * xi_n - xi_nm1);  // Mie-scattering coefficient b_n

    // Three-term Riccati-Bessel recurrence advances both function pairs from
    // orders n-1,n to n,n+1 without evaluating derivatives explicitly.
    T const psi_np1 = static_cast<T>(2 * n + 1) * psi_n / x - psi_nm1;
    auto const xi_np1 = static_cast<T>(2 * n + 1) * xi_n / x - xi_nm1;
    psi_nm1 = psi_n;
    psi_n = psi_np1;
    xi_nm1 = xi_n;
    xi_n = xi_np1;
  }

  T ext_sum = 0;
  T sca_sum = 0;
  T g_sum = 0;
  for (int n = 1; n <= nstop; ++n) {
    auto const an = a[n - 1];
    auto const bn = b[n - 1];
    T const weight = static_cast<T>(2 * n + 1);
    ext_sum += weight * (an.r + bn.r);  // Wiscombe 1980, eq.1a
    sca_sum +=
        weight * (complex_norm(an) + complex_norm(bn));  // Wiscombe 1980, eq.1b
    // The asymmetry-factor sum contains a_n*b_n^* and adjacent-order terms;
    // complex_conj() supplies the complex conjugate denoted by the star.
    g_sum += weight / static_cast<T>(n * (n + 1)) *
             (an * complex_conj(bn)).r;  // complex conjugate
    if (n < nstop) {
      auto const an1 = a[n];
      auto const bn1 = b[n];
      g_sum += static_cast<T>(n * (n + 2)) / static_cast<T>(n + 1) *
               (an * complex_conj(an1) + bn * complex_conj(bn1))
                   .r;  // Wiscombe 1980, eq.1c
    }
  }

  T const factor = static_cast<T>(2) / (x * x);
  out.qext = factor * ext_sum;
  out.qsca = factor * sca_sum;
  // Roundoff can make a nonabsorbing calculation very slightly violate
  // qext >= qsca. Enforce nonnegative absorption before forming the albedo.
  if (out.qext < out.qsca) out.qext = out.qsca;
  out.g = out.qsca > static_cast<T>(0)
              ? clamp_value(static_cast<T>(4) * g_sum / (x * x * out.qsca),
                            static_cast<T>(-0.999999), static_cast<T>(0.999999))
              : static_cast<T>(0);
  return out;
}

template <typename T>
DISPATCH_MACRO WaterLiquidMieProperties<T> water_liquid_mie_properties(
    T molar_conc, T wavelength, T radius_um, T density, T ref_real, T ref_imag,
    T molecular_weight, Complex<T>* work, int max_order) {
  if (molar_conc == static_cast<T>(0)) return {0, 0, 0};
  T const pi = static_cast<T>(3.141592653589793238462643383279502884);
  T const x = static_cast<T>(2) * pi * radius_um / wavelength;
  auto const mie =
      mie_efficiency_device(ref_real, ref_imag, x, work, max_order);
  if (mie.status != 0) {
    T const nan = static_cast<T>(NAN);
    return {nan, nan, nan};
  }

  T const radius_m = radius_um * static_cast<T>(1.0e-6);
  // kext = Qext*pi*r^2 / (4/3*rho_w*r^3) m^2/kg
  // More explicitly, the particle-mass denominator contains pi:
  // (4/3)*pi*rho_w*r^3. Cancelling pi and one power of r gives
  // 3*Qext/(4*rho_w*r).
  T const mass_extinction =
      static_cast<T>(3) * mie.qext / (static_cast<T>(4) * density * radius_m);
  T const extinction =  // m^-1
      molar_conc * molecular_weight * mass_extinction;
  T const single_scattering_albedo =
      mie.qext > static_cast<T>(0)
          ? clamp_value(mie.qsca / mie.qext, static_cast<T>(0),
                        static_cast<T>(1))
          : static_cast<T>(0);
  return {extinction, single_scattering_albedo, mie.g};
}

}  // namespace harp
