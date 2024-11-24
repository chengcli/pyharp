#pragma once

// C/C++
#include <future>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on
#include "radiation_band.hpp"

namespace harp {
extern std::unordered_map<std::string, std::shared_future<torch::Tensor>>
    shared;

struct RadiationOptions {
  RadiationOptions() = default;

  //! radiation input key in the input file
  ADD_ARG(std::string, input_key) = "radiation_config";
  ADD_ARG(bool, flux_flag) = false;
  ADD_ARG(bool, time_dependent) = false;
  ADD_ARG(bool, broad_band) = false;
  ADD_ARG(bool, stellar_beam) = false;
  ADD_ARG(bool, write_bin_radiance) = false;

  ADD_ARG(std::string, indirs) = "(0.,0.)";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::vector<std::string>, bands) = {};
  ADD_ARG(std::vector<RadiationBandOptions>, band_options) = {};

  void set_flags(std::string const& str);
};

class RadiationImpl : public torch::nn::Cloneable<RadiationImpl> {
 public:  // public access data
  //! options with which this `Radiation` was constructed
  RadiationOptions options;

  //! vertical coordinate
  torch::Tensor x1f;

  //! RadiationBands
  std::map<std::string, RadiationBand> bands;

  //! Constructor to initialize the layers
  RadiationImpl() = default;
  explicit RadiationImpl(RadiationOptions const& options_);
  void reset() override;

  //! \brief Calculate the radiance/radiative flux
  torch::Tensor forward(torch::Tensor ftoa, torch::Tensor var_x, float ray[2]);
};
TORCH_MODULE(Radiation);

}  // namespace harp
