#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace harp {
template <typename T>
bool real_close(T num1, T num2, T tolerance) {
  return std::fabs(num1 - num2) <= tolerance;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> get_direction_grids(
    std::vector<std::pair<T, T>> const &dirs) {
  std::vector<T> uphi;
  std::vector<T> umu;

  for (auto &dir : dirs) {
    // find phi
    bool found = false;
    for (auto &phi : uphi)
      if (real_close(phi, dir.phi, 1.e-3)) {
        found = true;
        break;
      }
    if (!found) uphi.push_back(dir.phi);

    // find mu
    found = false;
    for (auto &mu : umu)
      if (real_close(mu, dir.mu, 1.e-3)) {
        found = true;
        break;
      }
    if (!found) umu.push_back(dir.mu);
  }

  std::sort(uphi.begin(), uphi.end());
  std::sort(umu.begin(), umu.end());

  return std::make_pair(uphi, umu);
}
}  // namespace harp
