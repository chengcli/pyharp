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

struct S8RTOptions {
  static constexpr int npmom = 0;

  ADD_ARG(std::string, opacity_file) = "s8_k_fuller.txt";
  ADD_ARG(double, species_mu) = 256.e-3;  // [kg/mol]
  ADD_ARG(int, species_id) = 0;
  ADD_ARG(bool, use_wavenumber) = true;
};

class S8FullerImpl : public torch::nn::Cloneable<S8FullerImpl> {
 public:
  //! wavenumber [cm^-1] (default) or wavelength [um]
  //! (nwave, 1)
  torch::Tensor kwave;

  //! extinction x-section [m^2/mol] + single scattering albedo
  //! (nwave, nprop=2)
  torch::Tensor kdata;

  //! options with which this `S8FullerImpl` was constructed
  S8RTOptions options;

  //! Constructor to initialize the layer
  S8FullerImpl() = default;
  explicit S8FullerImpl(S8RTOptions const& options_);
  void reset() override;

  //! Get optical properties
  //! \param wave wavenumber [cm^-1], (nwave)
  //! \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
  //! \param pres pressure [Pa], (ncol, nlyr)
  //! \param temp temperature [K], (ncol, nlyr)
  torch::Tensor forward(torch::Tensor wave, torch::Tensor conc,
                        torch::optional<torch::Tensor> pres = torch::nullopt,
                        torch::optional<torch::Tensor> temp = torch::nullopt);
};
TORCH_MODULE(S8Fuller);

}  // namespace harp
