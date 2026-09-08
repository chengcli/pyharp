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
  torch::Tensor ln_sigma_binary, ln_temperature_base;

  double wavenumber_min = 0.0, wavenumber_max = 0.0;
  double pressure_min = 0.0, pressure_max = 0.0;
  double temperature_anomaly_min = 0.0, temperature_anomaly_max = 0.0;
  bool warned_wavenumber_bounds = false;
  bool warned_pressure_bounds = false;
  bool warned_temperature_anomaly_bounds = false;

  OpacityOptions options;

  MoleculeCIAImpl() : options(OpacityOptionsImpl::create()) {}
  explicit MoleculeCIAImpl(OpacityOptions const& options_);
  void reset() override;

  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(MoleculeCIA);

}  // namespace harp
