#pragma once

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

using RadiationBandOptionsMap = std::map<std::string, RadiationBandOptions>;

struct RadiationOptions {
  RadiationOptions() = default;

  //! radiation input key in the input file
  ADD_ARG(std::string, input_key) = "radiation_config";

  ADD_ARG(std::string, indirs) = "(0.,0.)";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(RadiationBandOptionsMap, band_options) = {};
};

class RadiationImpl : public torch::nn::Cloneable<RadiationImpl> {
 public:  // public access data
  //! surface dowward flux
  torch::Tensor downward;

  //! top of atmosphere upward flux
  torch::Tensor upward;

  //! all RadiationBands
  std::map<std::string, RadiationBand> bands;

  //! all band fluxes
  std::map<std::string, torch::Tensor> fluxes;

  //! options with which this `Radiation` was constructed
  RadiationOptions options;

  //! Constructor to initialize the layers
  RadiationImpl() = default;
  explicit RadiationImpl(RadiationOptions const& options_);
  void reset() override;

  //! \brief Calculate the radiance/radiative flux
  /*!
   * This function calls the forward function of each band and sums up the
   * fluxes. Once called, it will return the net flux and set internally
   * the surface downward flux and top of atmosphere upward flux.
   *
   * If kwargs contains "area", the fluxes will be scaled by
   * the spherical flux correction factor.
   * In this case, both "dz" and "vol" must be provided in kwargs as well
   *
   * \param conc mole concentration [mol/m^3] (ncol, nlyr, nspecies)
   * \param path layer pathlength (nlyr) or (ncol, nlyr)
   *
   * \param bc boundary conditions, may contain the following fields:
   *        "fbeam": solar beam irradiance [W/m^2], (nwave, ncol)
   *        "umu0": cosine of the solar zenith angle, (nwave, ncol)
   *        "albedo": surface albedo, (nwave, ncol)
   *
   * \param kwargs additional properties passed to other subroutines,
   *        may contain the following fields:
   *        "pres": pressure (ncol, nlyr)
   *        "temp": temperature (ncol, nlyr)
   *        "dz": layer thickness (nlyr) or (ncol, nlyr)
   *        "area": cell face area [m^2], (ncol)
   *        "vol": cell volume [m^3], (ncol)
   *
   * \return net flux tensor [W/m^2] (ncol, nlyr+1)
   */
  torch::Tensor forward(torch::Tensor conc, torch::Tensor path,
                        std::map<std::string, torch::Tensor>* bc,
                        std::map<std::string, torch::Tensor>* kwargs);
};
TORCH_MODULE(Radiation);

}  // namespace harp
