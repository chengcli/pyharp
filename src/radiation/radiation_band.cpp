// harp
#include "radiation_band.hpp"

#include <index.h>

#include <opacity/h2so4_simple.hpp>
#include <opacity/rfm.hpp>
#include <opacity/s8_fuller.hpp>
#include <utils/layer2level.hpp>
#include <utils/read_dimvar_netcdf.hpp>

#include "flux_utils.hpp"
#include "get_direction_grids.hpp"
#include "parse_radiation_direction.hpp"
#include "radiation.hpp"

namespace harp {

extern std::unordered_map<std::string, torch::Tensor> shared;

RadiationBandOptions RadiationBandOptions::from_yaml(
    std::string const& band_name, const YAML::Node& config) {
  RadiationBandOptions my;

  // band configuration
  auto band = config[band_name];

  my.name() = band_name;
  my.solver_name() = band["rt-solver"].as<std::string>();

  if (band["opacities"]) {
    for (auto const& name : band["opacities"]) {
      AttenuatorOptions a;
      a.name() = name.as<std::string>();
      a.opacity_files() = config[a.name()]["data"];
      replace_pattern_inplace(a.opacity_files(), "{band}", band_name);
      a.species_names() = config["species"];

      for (auto const& sp : config[a.name()]["species"]) {
        auto sp_name = sp.as<std::string>();
        // index sp_name in species
        auto it = std::find(a.species_names().begin(), a.species_names().end(),
                            sp_name);
        TORCH_CHECK(it != a.species_names().end(), "species ", sp_name,
                    " not found in species list");
        a.species_ids().push_back(it - a.species_names().begin());
      }
    }
  }

  int nwave = my.get_num_waves();
  auto [wmin, wmax] = get_wavenumber_range(band, nwave);

  if (my.solver_name() == "disort") {
    my.disort().header("running disort " + band_name);
    if (band["flags"]) {
      my.disort().flags() = band["flags"].as<std::string>();
    }
    my.disort().nwave(nwave);
    my.disort().wave_lower(std::vector<double>(nwave, wmin));
    my.disort().wave_upper(std::vector<double>(nwave, wmax));
  }

  if (band["ww"]) {
    my.ww() = band["ww"].as<std::vector<double>>();
  }
  my.integration() = band["integration"].as<std::string>();

  return my;
}

int RadiationBandOptions::get_num_waves() const {
  // user specified wave grid
  if (!ww().empty()) {
    return ww().size();
  }

  // cannot determine number of spectral grids if no opacities
  if (opacities().empty()) {
    TORCH_CHECK(false, "Unable to determine number of spectral grids");
  }

  // determine number of spectral grids from tabulated opacity sources
  auto op = opacities().begin()->second;
  auto wave = read_dimvar_netcdf<double>(op.opacity_files()[0], "Wavenumber");
  return wave.size();
}

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

  // create opacities
  for (auto const& [name, op] : options.opacities()) {
    if (op.type() == "rfm-lbl") {
      auto a = RFM(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
      options.ww() =
          read_dimvar_netcdf<double>(op.opacity_files()[0], "Wavenumber");
    } else if (op.type() == "rfm-ck") {
      auto a = RFM(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
      options.ww() =
          read_dimvar_netcdf<double>(op.opacity_files()[0], "weights");
    } else if (op.type() == "s8_fuller") {
      auto a = S8Fuller(op);
      nmax_prop_ = std::max(nmax_prop_, a->kdata.size(1));
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "h2sO4_simple") {
      auto a = H2SO4Simple(op);
      nmax_prop_ = std::max(nmax_prop_, a->kdata.size(1));
      opacities[name] = torch::nn::AnyModule(a);
    } else {
      TORCH_CHECK(false, "Unknown attenuator type: ", op.type());
    }
    register_module(name, opacities[name].ptr());
  }

  // create rtsolver
  auto [uphi, umu] = get_direction_grids<double>(ray_out);
  if (options.solver_name() == "disort") {
    rtsolver = torch::nn::AnyModule(disort::Disort(options.disort()));
    register_module("solver", rtsolver.ptr());
  } else {
    TORCH_CHECK(false, "Unknown solver: ", options.solver_name());
  }

  // create spectral grid
  ww = register_buffer("ww", torch::tensor(options.ww()));
}

torch::Tensor RadiationBandImpl::forward(
    torch::Tensor conc, torch::Tensor path,
    std::map<std::string, torch::Tensor>* bc,
    std::map<std::string, torch::Tensor>* kwargs) {
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  // add wavelength or wavenumber to kwargs, may overwrite existing values
  if (options.integration() == "wavenumber") {
    (*kwargs)["wavenumber"] = ww;
    (*kwargs)["wavelength"] = 1.e4 / ww;
  } else if (options.integration() == "wavelength") {
    (*kwargs)["wavenumber"] = 1.e4 / ww;
    (*kwargs)["wavelength"] = ww;
  }

  // bin optical properties
  auto prop =
      torch::zeros({ww.size(0), ncol, nlyr, nmax_prop_}, conc.options());

  for (auto& [_, a] : opacities) {
    auto kdata = a.forward(conc, *kwargs);
    int nprop = kdata.size(-1);

    // total extinction
    prop.select(-1, index::IEX) += kdata.select(-1, index::IEX);

    // single scattering albedo
    if (nprop > 1) {
      prop.select(-1, index::ISS) +=
          kdata.select(-1, index::ISS) * kdata.select(-1, index::IEX);
    }

    // phase moments
    if (nprop > 2) {
      prop.narrow(-1, index::IPM, nprop - 2) +=
          kdata.narrow(-1, index::IPM, nprop - 2) *
          kdata.select(-1, index::ISS) * kdata.select(-1, index::IEX);
    }
  }

  // extinction coefficients -> optical thickness
  int nprop = prop.size(-1);
  if (nprop > 2) {
    prop.narrow(-1, index::IPM, nprop - 2) /=
        (prop.select(-1, index::ISS) + 1e-10);
  }

  if (nprop > 1) {
    prop.select(-1, index::ISS) /= (prop.select(-1, index::IEX) + 1e-10);
  }

  prop.select(-1, index::IEX) *= path.unsqueeze(0);

  // export band optical properties
  std::string op_name = "radiation/" + options.name() + "/opacity";
  shared[op_name] = prop;

  std::string spec_name = "radiation/" + options.name() + "/spectra";

  // run rt solver
  if (kwargs->find("temp") != kwargs->end()) {
    Layer2LevelOptions l2l;
    l2l.order(k4thOrder).lower(kExtrapolate).upper(kConstant);
    shared[spec_name] = rtsolver.forward(
        prop, bc, std::make_optional(layer2level(kwargs->at("temp"), l2l)));
  } else {
    shared[spec_name] = rtsolver.forward(prop, bc);
  }

  // accumulate flux from flux spectra
  return cal_total_flux(shared[spec_name], ww, options.integration());
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
