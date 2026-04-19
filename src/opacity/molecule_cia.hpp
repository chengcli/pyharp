#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "opacity_options.hpp"

namespace harp {

class MoleculeCIAImpl : public torch::nn::Cloneable<MoleculeCIAImpl> {
 public:
  torch::Tensor wavenumber, ln_pressure, temperature_anomaly;
  torch::Tensor sigma_binary, temperature_base;

  OpacityOptions options;

  MoleculeCIAImpl() : options(OpacityOptionsImpl::create()) {}
  explicit MoleculeCIAImpl(OpacityOptions const& options_);
  void reset() override;

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(MoleculeCIA);

}  // namespace harp
