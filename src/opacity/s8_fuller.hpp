#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "attenuator_options.hpp"

namespace harp {

class S8FullerImpl : public torch::nn::Cloneable<S8FullerImpl> {
 public:
  //! wavelength [um]
  //! (nwave, 1)
  torch::Tensor kwave;

  //! extinction x-section [m^2/mol] + single scattering albedo
  //! (nwave, nprop=2)
  torch::Tensor kdata;

  //! options with which this `S8FullerImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  S8FullerImpl() = default;
  explicit S8FullerImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  /* \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
   *
   * \param kwargs arguments for opacity calculation, must contain:
   *        "wavelength": wavelength [um], (nwave)
   *
   * \return optical properties, (nwave, ncol, nlyr, nprop=2)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(S8Fuller);

}  // namespace harp
