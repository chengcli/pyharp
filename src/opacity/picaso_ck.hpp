#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "opacity_options.hpp"

namespace harp {

class PicasoCKImpl : public torch::nn::Cloneable<PicasoCKImpl> {
 public:
  torch::Tensor wavenumber, weights, wave_lower, wave_upper;
  torch::Tensor ln_pressure, temperature_anomaly, ln_temperature_base;
  torch::Tensor ln_sigma_cross;
  OpacityOptions options;

  PicasoCKImpl() : options(OpacityOptionsImpl::create()) {}
  explicit PicasoCKImpl(OpacityOptions const& options_);
  void reset() override;

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(PicasoCK);

}  // namespace harp
