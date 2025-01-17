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

struct H2SO4SimpleOptions {
  ADD_ARG(std::string, opacity_file) = "h2so4.txt";
  ADD_ARG(double, species_mu) = 100.e-3;  // [kg/mol]
  ADD_ARG(int, species_id) = 0;
  ADD_ARG(int, npmom) = 0;
};

class H2SO4SimpleImpl : public torch::nn::Cloneable<H2SO4SimpleImpl> {
 public:
  //! extinction x-section + single scattering albedo + phase function moments
  //! (nwave, nprop=3)
  torch::Tensor kdata;

  //! options with which this `H2SO4SimpleImpl` was constructed
  H2SO4SimpleOptions options;

  //! Constructor to initialize the layer
  H2SO4SimpleImpl() = default;
  explicit H2SO4SimpleImpl(H2SO4SimpleOptions const& options_);
  void reset() override;

  //! Get optical properties
  //! \param temp temperature [K], (ncol, nlyr)
  //! \param pres pressure [Pa], (ncol, nlyr)
  //! \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
  torch::Tensor forward(torch::Tensor conc,
                        torch::optional<torch::Tensor> pres = torch::nullopt,
                        torch::optional<torch::Tensor> temp = torch::nullopt);
};
TORCH_MODULE(H2SO4Simple);

}  // namespace harp
