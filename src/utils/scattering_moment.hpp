#pragma once

// torch
#include <torch/torch.h>

// clang-format off
#include <configure.h>
#include <add_arg.h>
#include <index.h>
// clang-format on

namespace harp {
struct PhaseMomentOptions {
  ADD_ARG(int, phase_func) = kRayleigh;
  ADD_ARG(float, gg) = 0.;
  ADD_ARG(float, gg1) = 0.;
  ADD_ARG(float, gg2) = 0.;
  ADD_ARG(float, ff) = 0.;
};

//! Compute the scattering phase moment
/*!
 * \param npmom Number of phase moments
 * \param options Options for the phase function
 * \return 1D tensor of phase moments, size = (1 + npmom,)
 */
torch::Tensor scattering_moment(int npmom, PhaseMomentOptions const &options);
}  // namespace harp
