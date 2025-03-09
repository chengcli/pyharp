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

  ADD_ARG(AttenuatorDict, opacity) = {};
  ADD_ARG(disort::DisortOptions, disort);

  ADD_ARG(std::vector<double>, ww) = { 1.0 };
  ADD_ARG(std::string, input) = "wavenumber";
};

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:
  //! spectral wavenumber, wavelength, or grid weights
  torch::Tensor ww;

  //! all opacities
  std::map<std::string, torch::nn::AnyModule> opacity;

  //! rt-solver
  torch::nn::AnyModule rtsolver;

  //! options with which this `RadiationBandImpl` was constructed
  RadiationBandOptions options;

  //! Constructor to initialize the layers
  RadiationBandImpl() = default;
  explicit RadiationBandImpl(RadiationBandOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& out) const override;

  torch::Tensor const& spectra() const { return spectra_; }
  torch::Tensor const& spectra() { return spectra_; }

  //! \brief Calculate the radiative flux
  /*!
   * This function calculates the cumulative radiative flux in the band
   * by summing up the fluxes of all spectral grids.
   * The spectral grids can be either line-by-line calculations
   * defined on wavenumber or wavelength grids or on a grid of correlated-k
   * points with weights.
   *
   * Once called, the flux spectra will be stored in the `spectra_` attribute
   * internally. `spectra_` is accessible by `spectra()` function.
   *
   * The last dimension of the returned flux tensor is 2, where the first
   * element is the upward flux and the second element is the downward flux.
   *
   * This function modifies the input `kwargs` by adding/modifiying the
   * wavelength or wavenumber grid to it.
   *
   * \param conc mole concentration [mol/m^3] (ncol, nlyr, nspecies)
   * \param path layer pathlength (nlyr) or (ncol, nlyr)
   *
   * \param bc boundary conditions, may contain the following fields:
   *        "fbeam": solar beam irradiance [W/m^2], (nwave, ncol)
   *        "umu0": cosine of the solar zenith angle, (nwave, ncol)
   *        "albedo": surface albedo, (nwave, ncol)
   *
   * \param kwargs other properties passed over to opacity claculations,
   *        may contain the following fields:
   *        "pres": pressure (ncol, nlyr)
   *        "temp": temperature (ncol, nlyr)
   *
   * \return cumulative radiative flux [W/m^2] (ncol, nlyr+1, 2)
   */
  torch::Tensor forward(torch::Tensor conc, torch::Tensor path,
                        torch::Tensor wave_or_weight, std::string input,
                        std::map<std::string, torch::Tensor>* bc,
                        std::map<std::string, torch::Tensor>* kwargs);

 private:
  //! radiance/flux spectra
  torch::Tensor spectra_;

  //! maximum number of optical property fields
  int64_t nmax_prop_ = 1;
};
TORCH_MODULE(RadiationBand);

}  // namespace harp
