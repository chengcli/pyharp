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

// Richard S. Freedman 2011. APJS
class FreedmanMeanImpl : public torch::nn::Cloneable<FreedmanMeanImpl> {
 public:
  static const double c1, c2, c3, c4, c5, c6, c7, c13;

  AttenuatorOptions options;

  FreedmanMeanImpl() = default;
  explicit FreedmanMeanImpl(AttenuatorOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(FreedmanMean);

// xiz semigrey
class SimpleGreyImpl : public torch::nn::Cloneable<SimpleGreyImpl> {
 public:
  AttenuatorOptions options;

  SimpleGreyImpl() = default;
  explicit SimpleGreyImpl(AttenuatorOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(SimpleGrey);

// xiz semigrey vis for Jupiter
class JupGasVisibleImpl : public torch::nn::Cloneable<JupGasVisibleImpl> {
 public:
  AttenuatorOptions options;

  JupGasVisibleImpl() = default;
  explicit JupGasVisibleImpl(AttenuatorOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(JupGasVisible);

// xiz semigrey ir for Jupiter
class JupGasIRImpl : public torch::nn::Cloneable<JupGasIRImpl> {
 public:
  AttenuatorOptions options;

  JupGasIRImpl() = default;
  explicit JupGasIRImpl(AttenuatorOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override {}
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(JupGasIR);

class GreyCloudImpl : public torch::nn::Cloneable<GreyCloudImpl> {
 public:
  AttenuatorOptions options;

  GreyCloudImpl() = default;
  explicit GreyCloudImpl(AttenuatorOptions const& options_);
  void reset() override {}

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(GreyCloud);

}  // namespace harp
