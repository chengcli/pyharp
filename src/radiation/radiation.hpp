#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "radiation_band.hpp"

// arg
#include <harp/add_arg.h>

namespace harp {

//! names of all species
extern std::vector<std::string> species_names;

//! molecular weights of all species [kg/mol]
extern std::vector<double> species_weights;

//! \brief Options for initializing a `Radiation` object
/*!
 * This class defines the options for initializing a `Radiation` object,
 * which is a module that calculates the cumulative radiative flux in
 * multiple spectral bands.
 *
 * The `RadiationOptions` object records the following options:
 *  - "outdirs": list of outgoing radiation directions
 *  - "band_options": dictionary of `RadiationBandOptions` objects
 *
 * The `RadiationOptions` object will store a `RadiationBandOptions` object
 * for each entry in the `band_options` dictionary.
 */
struct RadiationOptionsImpl {
  static std::shared_ptr<RadiationOptionsImpl> create() {
    return std::make_shared<RadiationOptionsImpl>();
  }

  //! \brief Create a `RadiationOptions` object from a YAML file
  /*!
   * This function reads a YAML file and creates a `RadiationOptions`
   * object from it. The YAML file must contain the following fields:
   *  - "species", list of species names and their composition
   *  - "bands": list of band configurations
   *
   * This function calls the `from_yaml` function of each `RadiationBandOptions`
   * object to create a `RadiationBandOptions` object for each band.
   */
  static std::shared_ptr<RadiationOptionsImpl> from_yaml(
      std::string const& filename);

  void report(std::ostream& os) const {
    os << "* outdirs = " << outdirs() << "\n";
    os << "* [ bands:\n";
    for (auto const& b : bands()) {
      os << "  - " << b->name() << ":\n";
      b->report(os);
    }
    os << "  ]\n";
  }

  int ncol() const {
    if (bands().empty()) return 0;
    int v = bands().front()->ncol();
    for (auto const& b : bands()) {
      TORCH_CHECK(b->ncol() == v, "number of columns in band ", b->name(), " (",
                  b->ncol(), ") does not match that of the first band (", v,
                  ")");
    }
    return v;
  }

  int nlyr() const {
    if (bands().empty()) return 0;
    int v = bands().front()->nlyr();
    for (auto const& b : bands()) {
      TORCH_CHECK(b->nlyr() == v, "number of layers in band ", b->name(), " (",
                  b->nlyr(), ") does not match that of the first band (", v,
                  ")");
    }
    return v;
  }

  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::vector<RadiationBandOptions>, bands);
};
using RadiationOptions = std::shared_ptr<RadiationOptionsImpl>;

class RadiationImpl : public torch::nn::Cloneable<RadiationImpl> {
 public:
  //! band spectra
  torch::Tensor spectra;

  //! all RadiationBands
  std::vector<RadiationBand> bands;

  //! options with which this `Radiation` was constructed
  RadiationOptions options;

  //! Constructor to initialize the layers
  RadiationImpl() : options(RadiationOptionsImpl::create()) {}
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
   * \param dz layer thickness (nlyr) or (ncol, nlyr)
   *
   * \param bc boundary conditions, may contain the following fields:
   *        <band> + "fbeam": solar beam irradiance [W/m^2], (nwave, ncol)
   *        <band> + "umu0": cosine of the solar zenith angle, (nwave, ncol)
   *        <band> + "albedo": surface albedo, (nwave, ncol)
   *        "btemp": bottom temperature
   *        "ttemp": top temperature
   *
   * \param kwargs additional properties passed to other subroutines,
   *        may contain the following fields:
   *        "pres": pressure (ncol, nlyr)
   *        "temp": temperature (ncol, nlyr)
   *        "area": cell face area [m^2], (ncol)
   *        "vol": cell volume [m^3], (ncol)
   *
   * \return (1) net flux tensor [W/m^2] (ncol, nlyr+1)
   *         (2) surface downward flux [W/m^2] (ncol)
   *         (3) TOA upward flux [W/m^2] (ncol)
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor conc, torch::Tensor dz,
      std::map<std::string, torch::Tensor>* bc,
      std::map<std::string, torch::Tensor>* kwargs);
};
TORCH_MODULE(Radiation);

}  // namespace harp

#undef ADD_ARG
