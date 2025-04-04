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

class HeliosImpl : public torch::nn::Cloneable<HeliosImpl> {
 public:
  //! data table coordinate axis
  //! (nband * ng,) (npres,) (ntemp,)
  torch::Tensor kwave, klnp, ktemp;

  //! g-point weights
  torch::Tensor weights;

  //! tabulated absorption x-section [ln(m^2/kmol)]
  //! (nwave, npres, ntemp, 1)
  torch::Tensor kdata;

  //! options with which this `HeliosImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  HeliosImpl() = default;
  explicit HeliosImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  /*!
   * This function calculates the absorption x-section of the gas
   * based on the tabulated data generated by the RFM.
   *
   * \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
   *
   * \param kwargs arguments for opacity calculation, must contain:
   *        "pres": pressure [Pa], (ncol, nlyr)
   *        "temp": temperature [K], (ncol, nlyr)
   *
   * \return optical properties, (nwave, ncol, nlyr, nprop=1)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(Helios);

}  // namespace harp
