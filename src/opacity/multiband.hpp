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

class MultiBandImpl : public torch::nn::Cloneable<MultiBandImpl> {
 public:
  //! data table coordinate axis
  //! (nband * ng,) (npres,) (ntemp,)
  torch::Tensor kwave, klnp, ktemp;

  //! g-point weights
  torch::Tensor weights;

  //! tabulated absorption x-section [ln(m^2/kmol)]
  //! (nwave, npres, ntemp, 1)
  torch::Tensor kdata;

  //! options with which this `MultiBandImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  MultiBandImpl() = default;
  explicit MultiBandImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  /*!
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
TORCH_MODULE(MultiBand);

}  // namespace harp
