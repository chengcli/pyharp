#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>
#include <torch/script.h>

// harp
#include "attenuator_options.hpp"

namespace harp {

class JITOpacityImpl : public torch::nn::Cloneable<JITOpacityImpl> {
 public:
  torch::jit::script::Module module;

  //! options with which this `JITOpacityImpl` was constructed
  AttenuatorOptions options;

  JITOpacityImpl() = default;
  explicit JITOpacityImpl(AttenuatorOptions const& options_);
  void reset() override;

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(JITOpacity);

}  // namespace harp
