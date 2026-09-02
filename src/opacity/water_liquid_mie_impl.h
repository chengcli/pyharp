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
  auto const a1 = static_cast<T>(2 * n + 1) / z;
  auto const a2 = -static_cast<T>(2 * n + 3) / z;
  auto u = a2 + static_cast<T>(1) / a1;
  auto v = a2;
  auto ratio = u / v;
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
  return static_cast<int>(
      x + static_cast<T>(4.05) * pow(x, static_cast<T>(1.0 / 3.0)) +
      static_cast<T>(2.0));
}

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

  int const computed_nstop = mie_nstop(x);
  int const nstop = computed_nstop > 1 ? computed_nstop : 1;
  int const derivative_order = nstop + 1;
  if (derivative_order + 1 > max_order) {
    out.status = 3;
    return out;
  }

  auto* d = work;
  auto* a = d + max_order;
  auto* b = a + max_order;
  auto const z = m * x;
  T const kappa = kimag;
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
  T psi_nm1 = sin_x;
  T psi_n = psi_nm1 / x - cos_x;
  Complex<T> xi_nm1(psi_nm1, cos_x);
  Complex<T> xi_n(psi_n, cos_x / x + sin_x);

  for (int n = 1; n <= nstop; ++n) {
    auto const dn = d[n];
    auto const nx = static_cast<T>(n) / x;
    auto const da = dn / m + nx;
    auto const db = m * dn + nx;
    a[n - 1] = (da * psi_n - psi_nm1) / (da * xi_n - xi_nm1);
    b[n - 1] = (db * psi_n - psi_nm1) / (db * xi_n - xi_nm1);

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
    ext_sum += weight * (an.r + bn.r);
    sca_sum += weight * (complex_norm(an) + complex_norm(bn));
    g_sum += weight / static_cast<T>(n * (n + 1)) * (an * complex_conj(bn)).r;
    if (n < nstop) {
      auto const an1 = a[n];
      auto const bn1 = b[n];
      g_sum += static_cast<T>(n * (n + 2)) / static_cast<T>(n + 1) *
               (an * complex_conj(an1) + bn * complex_conj(bn1)).r;
    }
  }

  T const factor = static_cast<T>(2) / (x * x);
  out.qext = factor * ext_sum;
  out.qsca = factor * sca_sum;
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
  T const mass_extinction =
      static_cast<T>(3) * mie.qext / (static_cast<T>(4) * density * radius_m);
  T const extinction = molar_conc * molecular_weight * mass_extinction;
  T const single_scattering_albedo =
      mie.qext > static_cast<T>(0)
          ? clamp_value(mie.qsca / mie.qext, static_cast<T>(0),
                        static_cast<T>(1))
          : static_cast<T>(0);
  return {extinction, single_scattering_albedo, mie.g};
}

}  // namespace harp
