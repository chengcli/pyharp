#pragma once

// C/C++
#include <algorithm>
#include <cmath>
#include <vector>

// torch
#include <torch/torch.h>

namespace harp {

//! \brief Check if two floating-point numbers are approximately equal
/*!
 * This function compares two floating-point numbers and returns true if
 * their absolute difference is within the specified tolerance.
 *
 * \tparam T The type of the numbers to compare (typically float or double)
 * \param num1 The first number to compare
 * \param num2 The second number to compare
 * \param tolerance The maximum allowed difference between the numbers
 * \return true if the numbers are within tolerance, false otherwise
 */
template <typename T>
bool real_close(T num1, T num2, T tolerance) {
  return std::fabs(num1 - num2) <= tolerance;
}

//! \brief Extract unique azimuthal and polar direction grids from a tensor
/*!
 * This function takes a tensor of direction pairs (phi, mu) and extracts
 * unique values for azimuthal angles (phi) and cosine of polar angles (mu).
 *
 * The input tensor `dirs` should be a 2D tensor with shape (n, 2), where:
 * - The first column contains azimuthal angles (phi)
 * - The second column contains cosine of polar angles (mu)
 *
 * Values are considered unique if they differ by more than 1e-3.
 * The returned vectors are sorted in ascending order.
 *
 * \tparam T The type of the direction values (typically float or double)
 * \param dirs A 2D tensor of shape (n, 2) containing direction pairs
 * \return A pair of vectors: (unique phi values, unique mu values)
 */
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
