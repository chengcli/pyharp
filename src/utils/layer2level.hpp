#pragma once

// torch
#include <torch/torch.h>

// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on

// harp
#include <index.h>

namespace harp {

constexpr int k2ndOrder = 2;
constexpr int k4thOrder = 4;

constexpr int kExtrapolate = 0;
constexpr int kConstant = 1;

struct Layer2LevelOptions {
  ADD_ARG(int, order) = k4thOrder;
  ADD_ARG(int, lower) = kExtrapolate;
  ADD_ARG(int, upper) = kConstant;
  ADD_ARG(bool, check_positivity) = true;
};

//! Convert layer variables to level variables
/*!
 * The layer variables are defined at the cell center, while the level variables
 * are defined at the cell interface. The last dimension of the input tensor is
 * the layer dimension.
 *
 * \param var layer variables, shape (..., nlayer)
 * \param options options
 * \return level variables, shape (..., nlevel = nlayer + 1)
 */
torch::Tensor layer2level(torch::Tensor var, Layer2LevelOptions const &options);
}  // namespace harp
