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
struct Layer2LevelOptions {
  ADD_ARG(int, order) = k4thOrder;
  ADD_ARG(bool, logx) = false;
  ADD_ARG(bool, logy) = false;
  ADD_ARG(int, blower) = kExtrapolate;
  ADD_ARG(int, bupper) = kConstant;
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
