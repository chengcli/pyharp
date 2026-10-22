#pragma once

// C/C++
#include <cmath>
#include <cstdint>

// disort
#include <disort/index.h>

// harp
#include <harp/math/rounding.h>

namespace harp {

// Every element of these kernels mirrors, operation for operation and in the
// same order, the tensor expressions the cloud modules evaluate when they
// cannot use the kernels, so both paths produce the same bits.

//! Layout of one wavelength's row in the Fu coefficient table.
namespace fu {
constexpr int kExt96 = 0;     // Fu96 extinction, 2 coefficients
constexpr int kCoalb96 = 2;   // Fu96 coalbedo, 4
constexpr int kAsym96 = 6;    // Fu96 asymmetry, 4
constexpr int kDelta96 = 10;  // Fu96 delta fraction, 4
constexpr int kExt98Lo = 14;  // Fu98 extinction at the lower node, 3
constexpr int kExt98Hi = 17;  // and at the upper node, 3
constexpr int kAbs98Lo = 20;  // Fu98 absorption, 4 + 4
constexpr int kAbs98Hi = 24;
constexpr int kAsym98Lo = 28;  // Fu98 asymmetry, 4 + 4
constexpr int kAsym98Hi = 32;
constexpr int kInterp = 36;     // Fu98 weight of the upper node
constexpr int kUseFu96 = 37;    // 1 where the lookup wavelength is below 4 um
constexpr int kSupported = 38;  // 1 within 0.25 to 100 um
constexpr int kWidth = 39;
}  // namespace fu

//! torch::minimum: a NaN operand is returned.
template <typename T>
DISPATCH_MACRO T nan_minimum(T a, T b) {
  if (a != a) return a;
  if (b != b) return b;
  return a < b ? a : b;
}

//! torch::clamp with scalar bounds: NaN passes through.
template <typename T>
DISPATCH_MACRO T nan_clamp(T x, T lo, T hi) {
  if (x != x) return x;
  return x < lo ? lo : (x > hi ? hi : x);
}

//! c0 + c1 d + c2 d^2 + c3 d^3, summed left to right.
template <typename T>
DISPATCH_MACRO T fu_cubic(T const* c, T d, T d2, T d3) {
  T s = rn_add(c[0], rn_mul(c[1], d));
  s = rn_add(s, rn_mul(c[2], d2));
  return rn_add(s, rn_mul(c[3], d3));
}

//! Fu ice optical properties of one cell at one wavelength.
/*!
 * \param out the cell's row (extinction, albedo, nmom phase moments)
 * \param row the wavelength's coefficient row, see namespace fu
 * \param d generalized effective size Dge [um]
 * \param iwc ice water content [g/m^3]
 * \return true when the properties are outside their physical ranges
 */
template <typename T>
DISPATCH_MACRO bool fu_ice_row(T* out, T const* row, T d, T iwc, int nmom,
                               bool delta_scale) {
  T const d2 = rn_mul(d, d);
  T const d3 = rn_mul(d2, d);

  // Fu96: piecewise-constant wavelength intervals.
  T ext96 = rn_add(row[fu::kExt96], rn_div(row[fu::kExt96 + 1], d));
  T ssa96 = rn_sub(T(1), fu_cubic(row + fu::kCoalb96, d, d2, d3));
  T g96 = fu_cubic(row + fu::kAsym96, d, d2, d3);
  if (delta_scale) {
    // Fu (1996), Eq. (3.8) and Appendix A.
    T f =
        rn_add(rn_div(T(0.5), ssa96), fu_cubic(row + fu::kDelta96, d, d2, d3));
    T const fw = rn_mul(f, ssa96);
    f = nan_minimum(f, g96);
    ext96 = rn_mul(rn_sub(T(1), fw), ext96);
    ssa96 = rn_div(rn_mul(rn_sub(T(1), f), ssa96), rn_sub(T(1), fw));
    g96 = rn_div(rn_sub(g96, f), rn_sub(T(1), f));
  }

  // Fu98: linear in wavelength between the surrounding nodes.
  auto ext98_at = [&](T const* c) {
    return rn_add(rn_add(c[0], rn_div(c[1], d)), rn_div(c[2], d2));
  };
  auto blend = [&](T lo, T hi) {
    return rn_add(lo, rn_mul(row[fu::kInterp], rn_sub(hi, lo)));
  };
  T const ext98 =
      blend(ext98_at(row + fu::kExt98Lo), ext98_at(row + fu::kExt98Hi));
  T const abs98 = blend(rn_div(fu_cubic(row + fu::kAbs98Lo, d, d2, d3), d),
                        rn_div(fu_cubic(row + fu::kAbs98Hi, d, d2, d3), d));
  T const g98 = blend(fu_cubic(row + fu::kAsym98Lo, d, d2, d3),
                      fu_cubic(row + fu::kAsym98Hi, d, d2, d3));
  T const ssa98 = rn_div(rn_sub(ext98, abs98), ext98);

  bool const use96 = row[fu::kUseFu96] != T(0);
  bool const supported = row[fu::kSupported] != T(0);
  T const ext = supported ? (use96 ? ext96 : ext98) : T(0);
  T const ssa = supported ? nan_clamp(use96 ? ssa96 : ssa98, T(0), T(1)) : T(0);
  T const g = supported
                  ? nan_clamp(use96 ? g96 : g98, T(-0.999999), T(0.999999))
                  : T(0);

  out[disort::IEX] = rn_mul(ext, iwc);
  out[disort::ISS] = ssa;
  // Henyey-Greenstein moments g^k. The device multiplies repeatedly, as
  // henyey_greenstein does on CUDA. On the host henyey_greenstein takes
  // torch::cumprod, whose CPU kernel accumulates float in double and rounds
  // each moment once, so do the same here.
#if defined(__CUDA_ARCH__)
  T moment = g;
  for (int k = 0; k < nmom; ++k) {
    if (k > 0) moment = rn_mul(moment, g);
#else
  using Acc = std::conditional_t<std::is_same<T, float>::value, double, T>;
  Acc product = Acc(1);
  for (int k = 0; k < nmom; ++k) {
    product *= static_cast<Acc>(g);
    T const moment = static_cast<T>(product);
#endif
    out[disort::IPM + k] = moment;
  }

  bool const finite_ext =
      ext == ext && ext != T(INFINITY) && ext != T(-INFINITY) && ext >= T(0);
  return !(finite_ext && ssa >= T(0) && ssa <= T(1) && g > T(-1) && g < T(1));
}

//! Combine ice and liquid optical properties of one cell as independent
//! constituents: extinction adds, the albedo is extinction weighted and the
//! phase moments are scattering weighted.
template <typename T>
DISPATCH_MACRO void cloud_switch_row(T* out, T const* ice, T const* liquid,
                                     int nmom) {
  T const ice_ext = ice[disort::IEX];
  T const liquid_ext = liquid[disort::IEX];
  T const ext = rn_add(ice_ext, liquid_ext);
  T const ice_sca = rn_mul(ice_ext, ice[disort::ISS]);
  T const liquid_sca = rn_mul(liquid_ext, liquid[disort::ISS]);
  T const sca = rn_add(ice_sca, liquid_sca);

  out[disort::IEX] = ext;
  out[disort::ISS] = ext > T(0) ? rn_div(sca, ext) : T(0);
  for (int k = 0; k < nmom; ++k) {
    T const weighted = rn_add(rn_mul(ice[disort::IPM + k], ice_sca),
                              rn_mul(liquid[disort::IPM + k], liquid_sca));
    out[disort::IPM + k] = sca > T(0) ? rn_div(weighted, sca) : T(0);
  }
}

}  // namespace harp
