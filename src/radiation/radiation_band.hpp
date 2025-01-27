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

#include <disort/disort.hpp>
#include <opacity/attenuator.hpp>
#include <rtsolver/rtsolver.hpp>
#include <utils/layer2level.hpp>

namespace harp {
struct RadiationBandOptions {
  RadiationBandOptions() = default;

  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::string, solver_name) = "lambert";

  ADD_ARG(std::vector<std::string>, attenuators) = {};
  ADD_ARG(std::vector<AttenuatorOptions>, attenuator_options) = {};
  ADD_ARG(Layer2LevelOptions, l2l);
  ADD_ARG(disort::DisortOptions, disort);

  // spectral dimension
  ADD_ARG(int, nprop) = 1;

  // atmosphere dimensions
  ADD_ARG(int, nlyr) = 1;
  ADD_ARG(int, ncol) = 1;

  //! set lower wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_lower) = {};

  //! set upper wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_upper) = {};
};

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:
  //! options with which this `RadiationBandImpl` was constructed
  RadiationBandOptions options;

  //! all attenuators
  std::map<std::string, Attenuator> attenuators;

  //! spectral grid weights
  //! (nwave)
  torch::Tensor weight;

  //! bin optical properties
  //! 5D tensor with shape (tau + ssa + pmom, C, ..., nlayer)
  torch::Tensor prop;

  //! outgoing rays (mu, phi)
  //! (nout, 2)
  torch::Tensor rayOutput;

  //! Constructor to initialize the layers
  RadiationBandImpl() = default;
  explicit RadiationBandImpl(RadiationBandOptions const &options_);
  void reset() override;
  std::string to_string() const;
  void load_opacity();

  //! \brief Calculate the radiance/radiative flux
  torch::Tensor forward(torch::Tensor x1f, torch::Tensor ftoa,
                        torch::Tensor var_x, double ray[2],
                        torch::optional<torch::Tensor> area = torch::nullopt,
                        torch::optional<torch::Tensor> vol = torch::nullopt);
};
TORCH_MODULE(RadiationBand);

/*class RadiationBandsFactory {
 public:
  static RadiationBandContainer CreateFrom(ParameterInput *pin,
                                           std::string key);
  static RadiationBandContainer CreateFrom(std::string filename);

  static int GetBandId(std::string const &bname) {
    if (band_id_.find(bname) == band_id_.end()) {
      return -1;
    }
    return band_id_.at(bname);
  }

 protected:
  static std::map<std::string, int> band_id_;
  static int next_band_id_;
};*/

}  // namespace harp
