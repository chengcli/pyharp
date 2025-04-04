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

class HydrogenCIAImpl : public torch::nn::Cloneable<HydrogenCIAImpl> {
 public:
  //! data table coordinate axis
  //! (nwave,) (ntemp,)
  torch::Tensor kwave, ktempa;

  //! extinction x-section + single scattering albedo + phase function moments
  //! (batch, specs, temps, levels, comps)
  torch::Tensor kdata_h2h2;
  torch::Tensor kdata_h2he;

  //! Constructor to initialize the layer
  HydrogenCIAImpl() = default;
  explicit HydrogenCIAImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(HydrogenCIA);

}  // namespace harp
