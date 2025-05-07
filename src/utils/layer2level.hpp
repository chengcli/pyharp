#pragma once

// torch
#include <torch/torch.h>

// harp
#include <harp/add_arg.h>

namespace harp {

constexpr int k2ndOrder = 2;
constexpr int k4thOrder = 4;

constexpr int kExtrapolate = 0;
constexpr int kConstant = 1;

struct Layer2LevelOptions {
  ADD_ARG(int, order) = k4thOrder;
  ADD_ARG(int, lower) = kExtrapolate;
  ADD_ARG(int, upper) = kConstant;
  ADD_ARG(bool, check_positivity) = false;
};

//! Convert layer variables to level variables for uniform mesh
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

//! Convert layer variables to level variables for non-uniform mesh
/*!
 * The layer variables are defined at the cell center, while the level variables
 * are defined at the cell interface. The last dimension of the input tensor is
 * the layer dimension.
 *
 * \param dx layer thickness, shape (..., nlayer)
 * \param var layer variables, shape (..., nlayer)
 * \param options options
 * \return level variables, shape (..., nlevel = nlayer + 1)
 */
torch::Tensor layer2level(torch::Tensor dx, torch::Tensor var,
                          Layer2LevelOptions const &options);

}  // namespace harp

#undef ADD_ARG
