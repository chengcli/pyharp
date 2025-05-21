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

class WaveTempImpl : public torch::nn::Cloneable<WaveTempImpl> {
 public:
  //! wavenumber coordinate and temperature
  //! (nwave,) (ntemp,)
  std::vector<torch::Tensor> kwave, ktemp;

  //! data table
  //! ncia x (nwave, ntemp)
  std::vector<torch::Tensor> kdata;

  //! options with which this `WaveTempImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  WaveTempImpl() = default;
  explicit WaveTempImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(WaveTemp);

}  // namespace harp
