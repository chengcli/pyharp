#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "add_arg.h"
#include "configure.h"
#include "radiation_band.hpp"

namespace harp {
struct RadiationOptions {
  RadiationOptions() = default;

  //! radiation input key in the input file
  ADD_ARG(std::string, input_key) = "radiation_config";
  ADD_ARG(bool, flux_flag) = false;
  ADD_ARG(bool, time_dependent) = false;
  ADD_ARG(bool, broad_band) = false;
  ADD_ARG(bool, thermal_emission) = false;
  ADD_ARG(bool, stellar_beam) = false;
  ADD_ARG(bool, normalize) = false;
  ADD_ARG(bool, write_bin_radiance) = false;
  ADD_ARG(bool, spectral_bin) = false;

  ADD_ARG(std::string, indirs) = "(0.,0.)";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::vector<std::string>, bands) = {};
  ADD_ARG(std::map<std::string, RadiationBandOptions>, band_options) = {};

  void set_flags(std::string const& str);
};

class RadiationImpl : public torch::nn::Cloneable<RadiationImpl> {
 public:  // public access data
  //! options with which this `Radiation` was constructed
  RadiationOptions options;

  //! vertical coordinate
  torch::Tensor x1f;

  //! incomming rays
  //! (nray, 2)
  torch::Tensor rayInput;

  //! RadiationBands
  std::map<std::string, RadiationBand> bands;

  //! Constructor to initialize the layers
  RadiationImpl() = default;
  explicit RadiationImpl(RadiationOptions const& options_);
  void reset() override;

  //! \brief Calculate the radiance/radiative flux
  torch::Tensor forward(torch::Tensor ftoa, torch::Tensor var_x);
};
TORCH_MODULE(Radiation);

}  // namespace harp
