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

class H2SO4SimpleImpl : public torch::nn::Cloneable<H2SO4SimpleImpl> {
 public:
  //! wavelength [um]
  //! (nwave, 1)
  torch::Tensor kwave;

  //! extinction x-section + single scattering albedo + phase function moments
  //! (nwave, nprop=2)
  torch::Tensor kdata;

  //! options with which this `H2SO4SimpleImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  H2SO4SimpleImpl() = default;
  explicit H2SO4SimpleImpl(AttenuatorOptions const& options_);
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
TORCH_MODULE(H2SO4Simple);

}  // namespace harp
