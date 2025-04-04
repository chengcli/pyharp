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

class GreyCloudImpl : public torch::nn::Cloneable<GreyCloudImpl> {
 public:
  AttenuatorOptions options;

  GreyCloudImpl() = default;
  explicit GreyCloudImpl(AttenuatorOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(GreyCloud);

}  // namespace harp
