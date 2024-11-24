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
  ADD_ARG(Layer2LevelOptions, l2l_options);
  ADD_ARG(DisortOptions, disort_options);

  ADD_ARG(int, nstr) = 1;
  ADD_ARG(int, nspec) = 1;
  ADD_ARG(int, nc1) = 1;
  ADD_ARG(int, nc2) = 1;
  ADD_ARG(int, nc3) = 1;

  ADD_ARG(float, wmin) = 0.0;
  ADD_ARG(float, wmax) = 1.0;
};

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:
  //! options with which this `RadiationBandImpl` was constructed
  RadiationBandOptions options;

  //! radiative transfer solver
  RTSolver solver;

  //! all attenuators
  std::map<std::string, Attenuator> attenuators;

  //! spectral grid and weights
  //! (2, nspec)
  torch::Tensor spec;

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
                        torch::Tensor var_x, float ray[2],
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
