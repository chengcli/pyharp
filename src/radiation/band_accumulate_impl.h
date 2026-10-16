#pragma once

// disort
#include <disort/index.h>

// harp
#include <harp/math/rounding.h>

namespace harp {

//! Add one opacity's row of optical properties into the band's row.
/*!
 * Mirrors, operation for operation, the tensor expressions the band used:
 *   prop[IEX] += k[IEX]
 *   prop[ISS] += k[ISS] * k[IEX]
 *   prop[IPM + m] += k[IPM + m] * (k[ISS] * k[IEX])
 * Both rows are contiguous along the property axis.
 */
template <typename T>
DISPATCH_MACRO void accumulate_opacity_row(T* prop, T const* k, int nprop) {
  T const ex = k[disort::IEX];
  prop[disort::IEX] = rn_add(prop[disort::IEX], ex);
  if (nprop > 1) {
    T const scattering = rn_mul(k[disort::ISS], ex);
    prop[disort::ISS] = rn_add(prop[disort::ISS], scattering);
    for (int m = 0; m < nprop - 2; ++m) {
      prop[disort::IPM + m] =
          rn_add(prop[disort::IPM + m], rn_mul(k[disort::IPM + m], scattering));
    }
  }
}

//! Turn the band's summed row into solver input.
/*!
 * In the order the band applied them: average the phase moments by the
 * summed scattering, average the scattering by the summed extinction (each
 * zero where the divisor is not positive), scale the extinction by the layer
 * thickness, and keep the albedo below one.
 */
template <typename T>
DISPATCH_MACRO void finish_band_row(T* prop, int nprop, T dz, T albedo_cap) {
  T const ex = prop[disort::IEX];
  if (nprop > 2) {
    T const ss = prop[disort::ISS];
    for (int m = 0; m < nprop - 2; ++m) {
      prop[disort::IPM + m] =
          ss > T(0) ? rn_div(prop[disort::IPM + m], ss) : T(0);
    }
  }
  if (nprop > 1) {
    T const ss = prop[disort::ISS];
    prop[disort::ISS] = ex > T(0) ? rn_div(ss, ex) : T(0);
  }
  prop[disort::IEX] = rn_mul(ex, dz);
  if (nprop > 1) {
    T const ss = prop[disort::ISS];
    prop[disort::ISS] = ss > albedo_cap ? albedo_cap : ss;
  }
}

}  // namespace harp
