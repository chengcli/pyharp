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
  //! wavenumber coordinate and temperature
  //! (nwave,) (ntemp,)
  torch::Tensor kwave, ktemp;

  //! data table
  //! (nwave, ntemp)
  torch::Tensor kdata;

  //! options with which this `HeliosImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  HydrogenCIAImpl() = default;
  explicit HydrogenCIAImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(HydrogenCIA);

class HydrogenHeliumCIAImpl
    : public torch::nn::Cloneable<HydrogenHeliumCIAImpl> {
 public:
  //! wavenumber coordinate and temperature
  //! (nwave,) (ntemp,)
  torch::Tensor kwave, ktemp;

  //! data table
  //! (nwave, ntemp)
  torch::Tensor kdata;

  //! options with which this `HeliosImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  HydrogenHeliumCIAImpl() = default;
  explicit HydrogenHeliumCIAImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(HydrogenHeliumCIA);

}  // namespace harp
