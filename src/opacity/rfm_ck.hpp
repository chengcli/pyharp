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
  constexpr static int IPR = 0;
  constexpr static int ITM = 1;

  //! ck table shape (nwave, npres, ntemp)
  size_t kshape[3];

  //! ck weights
  //! (nwave, 1)
  torch::Tensor kweight;

  //! ck table interpolation axis
  //! (nwave + npres + ntemp, 1)
  torch::Tensor kaxis;

  //! absorption x-section [m^2/mol]
  //! (nwave, nprop=1)
  torch::Tensor kdata;

  //! reference TP profile
  //! (2, npres)
  torch::Tensor krefatm;

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

torch::Tensor get_reftemp(torch::Tensor refatm, torch::Tensor lnp);

}  // namespace harp
