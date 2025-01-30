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

struct H2SO4RTOptions {
  static constexpr int npmom = 0;

  ADD_ARG(std::string, opacity_file) = "h2so4.txt";
  ADD_ARG(double, species_mu) = 98.e-3;  // [kg/mol]
  ADD_ARG(int, species_id) = 0;
  ADD_ARG(bool, use_wavenumber) = true;
};

class H2SO4SimpleImpl : public torch::nn::Cloneable<H2SO4SimpleImpl> {
 public:
  //! wavenumber [cm^-1] (default) or wavelength [um]
  //! (nwave, 1)
  torch::Tensor kwave;

  //! extinction x-section + single scattering albedo + phase function moments
  //! (nwave, nprop=3)
  torch::Tensor kdata;

  //! options with which this `H2SO4SimpleImpl` was constructed
  H2SO4RTOptions options;

  //! Constructor to initialize the layer
  H2SO4SimpleImpl() = default;
  explicit H2SO4SimpleImpl(H2SO4RTOptions const& options_);
  void reset() override;

  //! Get optical properties
  //! \param wave wavenumber [cm^-1], (nwave)
  //! \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
  //! \param pres pressure [Pa], (ncol, nlyr)
  //! \param temp temperature [K], (ncol, nlyr)
  //! \return optical properties, (nwave, ncol, nlyr, nprop)
  torch::Tensor forward(torch::Tensor wave, torch::Tensor conc,
                        torch::optional<torch::Tensor> pres = torch::nullopt,
                        torch::optional<torch::Tensor> temp = torch::nullopt);
};
TORCH_MODULE(H2SO4Simple);

}  // namespace harp
