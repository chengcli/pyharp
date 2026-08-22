#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "opacity_options.hpp"

namespace harp {

class RespqTableImpl : public torch::nn::Cloneable<RespqTableImpl> {
 public:
  torch::Tensor wavenumber, weights, wave_lower, wave_upper;
  torch::Tensor ln_pressure, temperature_anomaly, ln_temperature_base;
  torch::Tensor ln_linear, ln_binary, scattering, phase_moment;
  torch::Tensor reference_mole_fraction, bounds_mask;
  int nlinear = 0, nbinary = 0;

  OpacityOptions options;

  RespqTableImpl() : options(OpacityOptionsImpl::create()) {}
  explicit RespqTableImpl(OpacityOptions const& options_);
  void reset() override;
  int scattering_moments() const;

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(RespqTable);

}  // namespace harp
