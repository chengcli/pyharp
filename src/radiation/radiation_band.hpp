#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// yaml
#include <yaml-cpp/yaml.h>

// disort
#include <disort/disort.hpp>

// harp
#include <harp/opacity/attenuator_options.hpp>

// arg
#include <harp/add_arg.h>

namespace harp {

using AttenuatorDict = std::map<std::string, AttenuatorOptions>;
extern std::unordered_map<std::string, torch::Tensor> shared;

//! \brief Options for initializing a `RadiationBand` object
/*!
 * This class defines opacity sources and a radiative transfer solver
 * for a spectral band.
 *
 * It is used to initialize a `RadiationBand` object, which is a module
 * that calculates the cumulative radiative flux in the band.
 *
 * The `RadiationBand` object recognizes the following opacity source types:
 *  - "rfm-lbl": line-by-line opacity defined on wavenumber grid
 *  - "rfm-ck": correlated-k opacity table computed from rfm line-by-line table
 *  - "four-column": Four-column opacity table
 */
struct RadiationBandOptions {
  //! \brief Create a `RadiationBandOptions` object from a YAML file
  /*!
   * This function reads a YAML file and creates a `RadiationBandOptions`
   * object from it. The band node must contain the following fields:
   *  - "range": two float values defining the range of the band
   *  - "opacities": list of opacity sources
   *  - "solver": name of the radiative transfer solver
   *  - "integration": integration method ("wavenumber", "wavelength", or
   *                   "weight")
   *
   * It can optionally contain the following fields:
   *  - "flags": radiative transfer solver flags passed to the solver
   *
   * The returned `RadiationBandOptions` object will be partially filled
   * with the information from the YAML file. The YAML file does not
   * contain settings for the following variables:
   *  - number of spectral grids, default to 1
   *  - number of radiation streams, default to 4
   *  - number of columns, default to 1
   *  - number of layers, default to 1
   *
   * Thus, another call, such as, `get_num_waves`, is needed to set
   * these variables if they are different from the defaults.
   *
   * \param bd_name name of the band
   * \param config YAML configuration node
   * \return `RadiationBandOptions` object
   */
  static RadiationBandOptions from_yaml(std::string const& bd_name,
                                        YAML::Node const& config);

  //! \brief query the spectral grids
  /*!
   * This function queries the spectral grids of the first opacity source
   * An empty vector is returned if the subroutine cannot determine the
   * spectral grids.
   */
  std::vector<double> query_waves(std::string op_name = "") const;

  //! \brief query the spectral weights
  /*!
   * This function queries the spectral grid weights from the opacity source
   * An empty vector is returned if the subroutine cannot determine the
   * spectral weights.
   */
  std::vector<double> query_weights(std::string op_name = "") const;

  RadiationBandOptions() = default;

  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::string, solver_name) = "disort";

  ADD_ARG(AttenuatorDict, opacities) = {};
  ADD_ARG(disort::DisortOptions, disort);

  ADD_ARG(std::vector<double>, ww) = {};

  /// integration method: "wavenumber", "wavelength", or "weight"
  ADD_ARG(std::string, integration) = "wavenumber";
};

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:
  //! spectral wavenumber, wavelength, or grid weights
  torch::Tensor ww;

  //! all opacities
  std::map<std::string, torch::nn::AnyModule> opacities;

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
  /*!
   * This function calculates the cumulative radiative flux in the band
   * by summing up the fluxes of all spectral grids.
   *
   * The spectral grids can be either line-by-line calculations
   * defined on wavenumber or wavelength grids or on a grid of correlated-k
   * points with weights.
   *
   * The last dimension of the returned flux tensor is 2, where the first
   * element is the upward flux and the second element is the downward flux.
   *
   * This function modifies the input `kwargs` by adding/modifiying the
   * wavelength and wavenumber grid to it.
   *
   * This function exports the following tensor variables:
   *  - radiation/<band_name>/opacity
   *  - radiation/<band_name>/spectra
   *
   * \param conc mole concentration [mol/m^3] (ncol, nlyr, nspecies)
   * \param dz layer thickness (nlyr) or (ncol, nlyr)
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
  torch::Tensor forward(torch::Tensor conc, torch::Tensor dz,
                        std::map<std::string, torch::Tensor>* bc,
                        std::map<std::string, torch::Tensor>* kwargs);

 private:
  //! maximum number of optical property fields
  int64_t nmax_prop_ = 1;
};
TORCH_MODULE(RadiationBand);

}  // namespace harp

#undef ADD_ARG
