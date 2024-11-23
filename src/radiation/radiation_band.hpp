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

namespace harp {

using SharedData = std::shared_ptr<
    std::unordered_map<std::string, std::shared_future<torch::Tensor>>>;

struct RadiationBandOptions {
  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::string, solver) = "lambert";
  ADD_ARG(std::vector<std::string>, attenuators) = {};
  ADD_ARG(std::vector<AttenuatorOptions>, attenuator_options) = {};
  // ADD_ARG(SolverOptions, solver_options) = {};

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
  //! parameters for the model
  std::map<std::string, torch::Tensor> par;

  //! options with which this `RadiationBandImpl` was constructed
  RadiationBandOptions options;

  //! radiative transfer solver
  // RTSolver rt_solver;

  //! all attenuators
  std::map<std::string, Attenuator> attenuators;

  //! spectral grid and weights
  //! (nspec, 2)
  torch::Tensor spec;

  //! outgoing rays (mu, phi)
  //! (nout, 2)
  torch::Tensor rayOutput;

  //! band/bin optical data, 5D tensor with shape (3 + nstr, nc3, nc2, nc1)
  //! (tau + ssa + pmom, C, ..., nlayer)
  torch::Tensor opt;

  //! Constructor to initialize the layers
  RadiationBandImpl() = default;
  explicit RadiationBandImpl(RadiationBandOptions const &options_);
  void reset() override;

  //! \brief Calculate the radiance/radiative flux
  torch::Tensor forward(torch::Tensor x1f, torch::Tensor ftoa,
                        torch::Tensor var_x);

 protected:
  SharedData shared_;
  void set_temperature_level_(torch::Tensor hydro_x);
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
