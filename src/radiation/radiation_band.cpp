// yaml
#include <yaml-cpp/yaml.h>

// elements
#include <elements/utils.hpp>

// harp
#include <index.h>

#include <opacity/h2so4_simple.hpp>
#include <opacity/opacity_formatter.hpp>
#include <opacity/rfm.hpp>
#include <opacity/s8_fuller.hpp>
#include <utils/layer2level.hpp>
#include <utils/read_dimvar_netcdf.hpp>

#include "flux_utils.hpp"
#include "get_direction_grids.hpp"
#include "parse_radiation_direction.hpp"
#include "parse_yaml_input.hpp"
#include "radiation.hpp"
#include "radiation_band.hpp"
#include "radiation_formatter.hpp"

namespace harp {

extern std::unordered_map<std::string, torch::Tensor> shared;

RadiationBandOptions RadiationBandOptions::from_yaml(std::string const& bd_name,
                                                     const YAML::Node& config) {
  RadiationBandOptions my;

  // band configuration
  auto band = config[bd_name];
  TORCH_CHECK(band["opacities"], "opacities not found in band ", bd_name);

  for (auto const& op : band["opacities"]) {
    std::string op_name = op.as<std::string>();

    AttenuatorOptions a;
    a.bname(bd_name);

    TORCH_CHECK(config["opacities"][op_name], op_name,
                " not found in opacities");
    auto it = config["opacities"][op_name];

    TORCH_CHECK(it["type"], "'type' not found in opacity ", op_name);
    a.type(it["type"].as<std::string>());

    if (it["data"]) {
      a.opacity_files(it["data"].as<std::vector<std::string>>());
      for (auto& f : a.opacity_files()) {
        replace_pattern_inplace(f, "<band>", bd_name);
      }
    }

    if (it["species"]) {
      for (auto const& sp : it["species"]) {
        auto sp_name = sp.as<std::string>();

        // index sp_name in species
        auto jt =
            std::find(species_names.begin(), species_names.end(), sp_name);

        TORCH_CHECK(jt != species_names.end(), "species ", sp_name,
                    " not found in species list");
        a.species_ids().push_back(jt - species_names.begin());
      }
    }

    my.opacities()[op_name] = a;
  }

  auto [wmin, wmax] = parse_wave_range(band);

  my.name(bd_name);

  TORCH_CHECK(band["solver"], "'solver' not found in band ", bd_name);
  my.solver_name(band["solver"].as<std::string>());
  if (my.solver_name() == "disort") {
    my.disort().header("running disort " + bd_name);
    if (band["flags"]) {
      my.disort().flags(elements::trim_copy(band["flags"].as<std::string>()));
    }
    my.disort().nwave(1);
    my.disort().wave_lower(std::vector<double>(1, wmin));
    my.disort().wave_upper(std::vector<double>(1, wmax));
  } else if (my.solver_name() == "twostr") {
    TORCH_CHECK(false, "twostr solver not implemented");
  } else {
    TORCH_CHECK(false, "unknown solver: ", my.solver_name());
  }

  if (band["ww"]) {
    my.ww(band["ww"].as<std::vector<double>>());
  }

  TORCH_CHECK(band["integration"], "'integration' not found in band ", bd_name);
  my.integration(band["integration"].as<std::string>());

  return my;
}

std::vector<double> RadiationBandOptions::query_waves() const {
  // cannot determine number of spectral grids if no opacities
  if (opacities().empty()) {
    return {};
  }

  // determine number of spectral grids from tabulated opacity sources
  auto op = opacities().begin()->second;
  if (op.type().compare(0, 3, "rfm") == 0) {
    return read_dimvar_netcdf<double>(op.opacity_files()[0], "Wavenumber");
  } else {
    return {};
  }
}

std::vector<double> RadiationBandOptions::query_weights() const {
  // cannot determine number of spectral grids if no opacities
  if (opacities().empty()) {
    return {};
  }

  // determine number of spectral grids from tabulated opacity sources
  auto op = opacities().begin()->second;
  if (op.type().compare(0, 3, "rfm") == 0) {
    return read_dimvar_netcdf<double>(op.opacity_files()[0], "weights");
  } else {
    return {};
  }
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
    } else if (op.type() == "h2so4_simple") {
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
  ww = register_buffer("ww", torch::tensor(options.ww(), torch::kFloat64));
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
    l2l.order(k4thOrder).lower(kExtrapolate).upper(kExtrapolate);
    shared[spec_name] = rtsolver.forward(
        prop, bc, options.name(),
        std::make_optional(layer2level(kwargs->at("temp"), l2l)));
  } else {
    shared[spec_name] = rtsolver.forward(prop, bc, options.name());
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
