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

class WaveTempTableImpl : public torch::nn::Cloneable<WaveTempTableImpl> {
 public:
  //! wavenumber coordinate and temperature
  //! (nwave,) (ntemp,)
  torch::Tensor kwave, ktemp;

  //! data table
  //! (ncia, nwave, ntemp)
  torch::Tensor kdata;

  //! options with which this `WaveTempTableImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  WaveTempTableImpl() = default;
  explicit WaveTempTableImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(WaveTempTable);

}  // namespace harp
