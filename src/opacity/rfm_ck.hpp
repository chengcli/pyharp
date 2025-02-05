#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <add_arg.h>

#include "attenuator.hpp"

namespace harp {

class RFMCKImpl : public torch::nn::Cloneable<RFMCKImpl> {
 public:
  //! shape of interpolation axes, ntemp, npres, ngpoints
  size_t len[3];

  //! ck weights
  //! (nwave, 1)
  torch::Tensor kweight;

  //! ck table interpolation axis
  //! (nwave, 1)
  torch::Tensor kaxis;

  //! absorption x-section [m^2/mol]
  //! (nwave, nprop=1)
  torch::Tensor kdata;

  //! options with which this `RFMCKImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  RFMCKImpl() = default;
  explicit RFMCKImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  //! \param xfrac gas mole fraction [1], (ncol, nlyr, nspecies)
  //! \param pres pressure [Pa], (ncol, nlyr)
  //! \param temp temperature [K], (ncol, nlyr)
  //! \return optical properties, (nwave, ncol, nlyr, nprop=1)
  torch::Tensor forward(torch::Tensor xfrac, torch::Tensor pres,
                        torch::Tensor temp);
};
TORCH_MODULE(RFMCK);

}  // namespace harp
