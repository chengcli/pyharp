#pragma once

#include <cmath>
#include <string>

namespace at {
class Tensor;
}

namespace torch {
using Tensor = at::Tensor;
}

namespace harp {
template <typename T>
T rad2deg(T phi) {
  return phi * 180. / M_PI;
}

template <typename T>
T deg2rad(T phi) {
  return phi * M_PI / 180.;
}

torch::Tensor parse_radiation_direction(std::string const &str);
torch::Tensor parse_radiation_directions(std::string const &str);
}  // namespace harp
