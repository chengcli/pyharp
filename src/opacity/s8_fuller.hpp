#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on

namespace harp {

struct S8FullerOptions {
  ADD_ARG(std::string, opacity_file) = "s8_k_fuller.txt";
  ADD_ARG(double, species_mu) = 256.e-3;  // [kg/mol]
  ADD_ARG(int, species_id) = 0;
};

class S8FullerImpl : public torch::nn::Cloneable<S8FullerImpl> {
 public:
  //! extinction x-section + single scattering albedo + phase function moments
  //! (nwave, nprop=3)
  torch::Tensor kdata;

  //! options with which this `S8FullerImpl` was constructed
  S8FullerOptions options;

  //! Constructor to initialize the layer
  S8FullerImpl() = default;
  explicit S8FullerImpl(S8FullerOptions const& options_);
  void reset() override;

  //! Get optical properties
  //! \param temp temperature [K], (ncol, nlyr)
  //! \param pres pressure [Pa], (ncol, nlyr)
  //! \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor conc);
};
TORCH_MODULE(S8Fuller);

}  // namespace harp
