#pragma once

// C/C++
#include <algorithm>
#include <cmath>
#include <vector>

// torch
#include <torch/torch.h>

namespace harp {
template <typename T>
bool real_close(T num1, T num2, T tolerance) {
  return std::fabs(num1 - num2) <= tolerance;
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> get_direction_grids(
    torch::Tensor dirs) {
  std::vector<T> uphi;
  std::vector<T> umu;

  for (int i = 0; i < dirs.size(0); ++i) {
    // find phi
    bool found = false;
    for (auto &phi : uphi)
      if (real_close(phi, dirs[i][0].item<T>(), 1.e-3)) {
        found = true;
        break;
      }
    if (!found) uphi.push_back(dirs[i][0].item<T>());

    // find mu
    found = false;
    for (auto &mu : umu)
      if (real_close(mu, dirs[i][1].item<T>(), 1.e-3)) {
        found = true;
        break;
      }
    if (!found) umu.push_back(dirs[i][1].item<T>());
  }

  std::sort(uphi.begin(), uphi.end());
  std::sort(umu.begin(), umu.end());

  return std::make_pair(uphi, umu);
}
}  // namespace harp
