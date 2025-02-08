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

class RFMImpl : public torch::nn::Cloneable<RFMImpl> {
 public:
  constexpr static int IPR = 0;
  constexpr static int ITM = 1;

  //! data table shape (nwave, npres, ntemp)
  size_t kshape[3];

  //! data table interpolation axis
  //! (nwave + npres + ntemp,)
  torch::Tensor kaxis;

  //! tabulated absorption x-section [ln(m^2/kmol)]
  //! (nwave, npres, ntemp)
  torch::Tensor kdata;

  //! reference TP profile (lnp, temp)
  //! (2, npres)
  torch::Tensor krefatm;

  //! options with which this `RFMImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  RFMImpl() = default;
  explicit RFMImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  /* \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
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
TORCH_MODULE(RFM);

torch::Tensor get_reftemp(torch::Tensor lnp, torch::Tensor klnp,
                          torch::Tensor ktemp);

}  // namespace harp
