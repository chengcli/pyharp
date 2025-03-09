// C/C++
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// harp
#include <opacity/h2so4_simple.hpp>
#include <opacity/rfm.hpp>
#include <opacity/s8_fuller.hpp>
#include <utils/get_direction_grids.hpp>
#include <utils/parse_radiation_direction.hpp>
#include <utils/spherical_flux_correction.hpp>

#include "flux_utils.hpp"
#include "radiation.hpp"
#include "radiation_band.hpp"

namespace harp {
RadiationBandImpl::RadiationBandImpl(RadiationBandOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationBandImpl::reset() {
  auto str = options.outdirs();
  torch::Tensor ray_out;
  if (!str.empty()) {
    ray_out = parse_radiation_directions(str);
  }

  // create spectral grid
  ww = register_buffer("ww", torch::tensor(options.ww()));

  // create opacities
  for (auto const& [name, op] : options.opacities()) {
    if (op.type() == "rfm") {
      auto a = RFM(op);
      nmax_prop_ = std::max(nmax_prop_, a->kdata.size(1));
      opacity[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "s8_fuller") {
      auto a = S8Fuller(op);
      nmax_prop_ = std::max(nmax_prop_, a->kdata.size(1));
      opacity[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "h2sO4_simple") {
      auto a = H2SO4Simple(op);
      nmax_prop_ = std::max(nmax_prop_, a->kdata.size(1));
      opacity[name] = torch::nn::AnyModule(a);
    } else {
      TORCH_CHECK(false, "Unknown attenuator type: ", op.type());
    }
    register_module(name, opacity[name].ptr());
  }

  // create rtsolver
  auto [uphi, umu] = get_direction_grids<double>(ray_out);
  if (options.solver_name() == "disort") {
    rtsolver = torch::nn::AnyModule(disort::Disort(options.disort()));
    register_module("solver", rtsolver.ptr());
  } else {
    TORCH_CHECK(false, "Unknown solver: ", options.solver_name());
  }
}

torch::Tensor RadiationBandImpl::forward(
    torch::Tensor conc, torch::Tensor dz,
    std::map<std::string, torch::Tensor>* bc,
    std::map<std::string, torch::Tensor>* kwargs) {
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  // add wavelength or wavenumber to kwargs, may overwrite existing values
  if (options.input() == "wavenumber") {
    kwargs["wavenumber"] = ww;
    kwargs["wavelength"] = 1.e4 / ww;
  } else if (options.input() == "wavelength") {
    kwargs["wavenumber"] = 1.e4 / ww;
    kwargs["wavelength"] = ww;
  }

  // bin optical properties
  auto prop = torch::zeros({nmax_prop_, ncol, nlyr}, conc.options());

  for (auto& [_, a] : opacity) {
    auto kdata = a.forward(conc, *kwargs);
    int nprop = kdata.size(0);

    // total extinction
    prop[index::IEX] += kdata[index::IEX];

    // single scattering albedo
    if (nprop > 1) {
      prop[index::ISS] += kdata[index::ISS] * kdata[index::IEX];
    }

    // phase moments
    if (nprop > 2) {
      prop.narrow(0, index::IPM, nprop - 2) +=
          kdata.narrow(0, index::IPM, nprop - 2) * kdata[index::ISS] *
          kdata[index::IEX];
    }
  }

  // extinction coefficients -> optical thickness
  int nprop = prop.size(0);
  if (nprop > 2) {
    prop.narrow(0, index::IPM, nprop - 2) /= (prop[index::ISS] + 1e-10);
  }

  if (nprop > 1) {
    prop[index::ISS] /= (prop[index::IEX] + 1e-10);
  }

  prop[index::IEX] *= dz.unsqueeze(0);

  // export band optical properties
  std::string name = "radiation/" + options.name() + "/optics";
  shared[name] = prop;

  // run rt solver
  if (kwargs.count("temp") > 0) {
    Layer2LevelOptions l2l;
    l2l.order(k4thOrder).lower(kExtrapolate).upper(kConstant);
    spectra_ = rtsolver.forward(prop, bc, layer2level(kwargs.at("temp"), l2l));
  } else {
    spectra_ = rtsolver.forward(prop, bc);
  }

  // accumulate flux from flux spectra
  return cal_total_flux(spectra_, ww, options.input());
}

void RadiationBandImpl::pretty_print(std::ostream& out) const {
  out << "RadiationBand: " << options.name() << std::endl;
  out << "Absorbers: (";
  for (auto const& [name, _] : opacities) {
    out << name << ", ";
  }
  out << ")" << std::endl;
  out << std::endl << "Solver: " << options.solver_name();
}

}  // namespace harp
