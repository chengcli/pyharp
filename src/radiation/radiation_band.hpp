#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// disort
#include <disort/disort.hpp>

// harp
#include <harp/harp_formatter.hpp>
#include <harp/opacity/opacity_options.hpp>
#include <harp/rtsolver/toon_mckay89.hpp>

// arg
#include <harp/add_arg.h>

namespace harp {

using OpacityDict = std::map<std::string, OpacityOptions>;

//! \brief Options for initializing a `RadiationBand` object
/*!
 * This class defines opacity sources and a radiative transfer solver
 * for a spectral band.
 *
 * It is used to initialize a `RadiationBand` object, which is a module
 * that calculates the cumulative radiative flux in the band.
 *
 * The `RadiationBand` object recognizes the following opacity source types:
 *  - "jit": user-defined opacity module
 *  - "rfm-lbl": line-by-line opacity defined on wavenumber grid
 *  - "rfm-ck": correlated-k opacity table computed from rfm line-by-line table
 *  - "multiband-ck": multi-band correlated-k opacity table
 *  - "wavetemp": opacity table defined on wavenumber and temperature grid (CIA)
 *  - "fourcolumn": Four-column opacity table (aerosol)
 *  - "helios": Helios opacity table
 */
struct RadiationBandOptionsImpl {
  static std::shared_ptr<RadiationBandOptionsImpl> create() {
    return std::make_shared<RadiationBandOptionsImpl>();
  }

  //! \brief Create a `RadiationBandOptions` object from a YAML file
  /*!
   * This function reads a YAML file and creates a `RadiationBandOptions`
   * object from it. The band node must contain the following fields:
   *  - "range": two float values defining the range of the band
   *  - "opacities": list of opacity sources
   *  - "solver": name of the radiative transfer solver
   *
   * It can optionally contain the following fields:
   *  - "flags": radiative transfer solver flags passed to the solver
   *
   * The returned `RadiationBandOptions` object will be partially filled
   * with the information from the YAML file.
   *
   * Another call to set wavenumber or weight
   * is needed to make it a complete `RadiationBandOptions` object.
   *
   * \param filename name of the YAML file
   * \param bd_name name of the band
   * \return `RadiationBandOptions` object
   */
  static std::shared_ptr<RadiationBandOptionsImpl> from_yaml(
      std::string const& filename, std::string const& bd_name);

  std::shared_ptr<RadiationBandOptionsImpl> clone() const {
    auto op = std::make_shared<RadiationBandOptionsImpl>(*this);
    if (op->disort() != nullptr) {
      op->disort() = op->disort()->clone();
    }
    if (op->toon() != nullptr) {
      op->toon() = op->toon()->clone();
    }
    if (op->opacities().size() > 0) {
      auto& opacities_ref = op->opacities();
      opacities_ref.clear();
      for (auto const& [k, v] : opacities()) {
        opacities_ref[k] = v->clone();
      }
    }
    return op;
  }
  void report(std::ostream& os) const {
    os << "* name = " << name() << "\n";
    os << "* outdirs = " << outdirs() << "\n";
    os << "* solver_name = " << solver_name() << "\n";
    os << "* l2l_order = " << l2l_order() << "\n";
    os << "* nwave = " << nwave() << "\n";
    os << "* ncol = " << ncol() << "\n";
    os << "* nlyr = " << nlyr() << "\n";
    os << "* nstr = " << nstr() << "\n";

    os << "* opacities: [\n";
    for (auto const& [k, v] : opacities()) {
      os << "  - " << k << ":\n";
      v->report(os);
    }
    os << "  ]\n";

    if (solver_name() == "disort") {
      os << "* disort:\n";
      disort()->report(os);
    } else if (solver_name() == "toon") {
      os << "* toon:\n";
      toon()->report(os);
    }

    os << "* wavenumber = " << fmt::format("{}", wavenumber()) << "\n";
    os << "* weight = " << fmt::format("{}", weight()) << "\n";
    os << "* verbose = " << (verbose() ? "true" : "false") << "\n";
  }

  ADD_ARG(std::string, name) = "B1";
  ADD_ARG(std::string, outdirs) = "";
  ADD_ARG(std::string, solver_name) = "disort";
  ADD_ARG(int, l2l_order) = 2;

  ADD_ARG(int, nwave) = 0;
  ADD_ARG(int, ncol) = 0;
  ADD_ARG(int, nlyr) = 0;
  ADD_ARG(int, nstr) = 4;

  ADD_ARG(OpacityDict, opacities) = {};
  ADD_ARG(disort::DisortOptions, disort);
  ADD_ARG(ToonMcKay89Options, toon);

  ADD_ARG(std::vector<double>, wavenumber);
  ADD_ARG(std::vector<double>, weight);
  ADD_ARG(bool, verbose) = false;
};
using RadiationBandOptions = std::shared_ptr<RadiationBandOptionsImpl>;

class RadiationBandImpl : public torch::nn::Cloneable<RadiationBandImpl> {
 public:
  //! bin optical properties
  torch::Tensor prop;

  //! bin spectrum
  torch::Tensor spectrum;

  //! all opacities
  std::map<std::string, torch::nn::AnyModule> opacities;

  //! rt-solver
  torch::nn::AnyModule rtsolver;

  //! options with which this `RadiationBandImpl` was constructed
  RadiationBandOptions options;

  //! Constructor to initialize the layers
  RadiationBandImpl() : options(RadiationBandOptionsImpl::create()) {}
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
};
TORCH_MODULE(RadiationBand);

}  // namespace harp

#undef ADD_ARG
