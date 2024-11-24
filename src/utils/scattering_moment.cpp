// C/C++
#include <cmath>

// cdisort
#include <rtsolver/cdisort213/cdisort.h>  // c_getmom

// conflict with torch
#undef A
#undef B

// harp
#include "scattering_moment.hpp"

namespace harp {
torch::Tensor scattering_moment(int npmom, PhaseMomentOptions const &options) {
  torch::Tensor pmom = torch::zeros({1 + npmom}, torch::kDouble);

  if (options.phase_func() == kHenyeyGreenstein) {
    if (options.gg() <= -1. || options.gg() >= 1.) {
      throw std::runtime_error("scattering_moment::bad input variable gg");
    }
    for (int k = 1; k <= npmom; k++) {
      pmom[k] = pow(options.gg(), (float)k);
    }
  } else if (options.phase_func() == kDoubleHenyeyGreenstein) {
    auto gg1 = options.gg1();
    auto gg2 = options.gg2();
    auto ff = options.ff();

    if (gg1 <= -1. || gg1 >= 1. || gg2 <= -1. || gg2 >= 1.) {
      throw std::runtime_error(
          "get_phase_moment::bad input variable gg1 or gg2");
    }

    for (int k = 1; k <= npmom; k++) {
      pmom[k] = ff * pow(gg1, (float)k) + (1. - ff) * pow(gg2, (float)k);
    }
  } else if (options.phase_func() == kRayleigh) {
    if (npmom < 2) {
      throw std::runtime_error("scattering_moment::npmom < 2");
    }
    pmom[2] = 0.1;
  } else if (options.phase_func() == kHazeGarciaSiewert) {
    c_getmom(HAZE_GARCIA_SIEWERT, options.gg(), npmom, pmom.data_ptr<double>());
  } else if (options.phase_func() == kCloudGarciaSiewert) {
    c_getmom(CLOUD_GARCIA_SIEWERT, options.gg(), npmom,
             pmom.data_ptr<double>());
  } else {
    throw std::runtime_error("scattering_moment::unknown phase function");
  }

  return pmom;
}
}  // namespace harp
