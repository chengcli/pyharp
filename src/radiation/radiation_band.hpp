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
#include <opacity/attenuator_options.hpp>
#include <rtsolver/rtsolver.hpp>
#include <utils/layer2level.hpp>

namespace harp {
using AttenuatorDict = std::map<std::string, AttenuatorOptions>;

struct RadiationBandOptions {
  RadiationBandOptions() = default;

  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::string, solver_name) = "disort";

  ADD_ARG(AttenuatorDict, attenuators) = {};
  ADD_ARG(Layer2LevelOptions, l2l);
  ADD_ARG(disort::DisortOptions, disort);

  //! atmospheric layers
  ADD_ARG(int, nlyr) = 1;

  //! atmospheric columns
  ADD_ARG(int, ncol) = 1;

  //! optical dimension
  ADD_ARG(int64_t, nmax_prop) = 1;

  //! set lower wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_lower) = {};

  //! set upper wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_upper) = {};
};

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:
  //! all attenuators
  std::map<std::string, torch::nn::AnyModule> attenuators;

  //! rt-solver
  torch::nn::AnyModule rtsolver;

  //! options with which this `RadiationBandImpl` was constructed
  RadiationBandOptions options;

  //! Constructor to initialize the layers
  RadiationBandImpl() = default;
  explicit RadiationBandImpl(RadiationBandOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& out) const override;

  //! \brief Calculate the radiative flux
  /* \param conc mole concentration [mol/m^3] (ncol, nlyr, nspecies)
   * \param dz layer thickness (nlyr)
   *
   * \param bc boundary conditions, may contain the following fields:
   *        "fbeam": solar beam irradiance [W/m^2], (nwave, ncol)
   *        "umu0": cosine of the solar zenith angle, (nwave, ncol)
   *        "albedo": surface albedo, (nwave, ncol)
   *
   * \param op optics properties, may contain the following fields:
   *        "wavelength": wavelength [um], (nwave)
   *        "wavenumber": wavenumber [cm^-1], (nwave)
   *        "pres": pressure (ncol, nlyr)
   *        "temp": temperature (ncol, nlyr)
   */
  torch::Tensor forward(torch::Tensor conc, torch::Tensor dz,
                        std::map<std::string, torch::Tensor>& bc,
                        std::map<std::string, torch::Tensor> const& op);
};
TORCH_MODULE(RadiationBand);

}  // namespace harp
