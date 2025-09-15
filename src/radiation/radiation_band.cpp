// yaml
#include <yaml-cpp/yaml.h>

// harp
#include <harp/index.h>

#include <harp/opacity/fourcolumn.hpp>
#include <harp/opacity/grey_opacities.hpp>
#include <harp/opacity/helios.hpp>
#include <harp/opacity/jit_opacity.hpp>
#include <harp/opacity/multiband.hpp>
#include <harp/opacity/opacity_formatter.hpp>
#include <harp/opacity/rfm.hpp>
#include <harp/opacity/wavetemp.hpp>
#include <harp/utils/layer2level.hpp>
#include <harp/utils/read_dimvar_netcdf.hpp>
#include <harp/utils/read_var_pt.hpp>
#include <harp/utils/strings.hpp>

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

    if (it["nmom"]) {
      a.nmom(it["nmom"].as<int>());
    }

    if (it["fractions"]) {
      a.fractions(it["fractions"].as<std::vector<double>>());
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
      my.disort().flags(trim_copy(band["flags"].as<std::string>()));
    }
    my.disort().nwave(1);
    my.disort().upward(true);
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

std::vector<double> RadiationBandOptions::query_waves(
    std::string op_name) const {
  // assign first opacity if no name is given
  if (op_name.empty()) {
    op_name = opacities().begin()->first;
  }

  // cannot determine spectral grids if no opacities
  if (opacities().empty() || opacities().find(op_name) == opacities().end()) {
    return {};
  }

  auto op = opacities().at(op_name);
  if (op.type().compare(0, 3, "rfm") == 0) {
    return read_dimvar_netcdf<double>(op.opacity_files()[0], "Wavenumber");
  } else if (op.type().compare(0, 9, "multiband") == 0) {
    return read_var_pt<double>(op.opacity_files()[0], "wavenumber");
  } else {
    return {};
  }
}

std::vector<double> RadiationBandOptions::query_weights(
    std::string op_name) const {
  // assign first opacity if no name is given
  if (op_name.empty()) {
    op_name = opacities().begin()->first;
  }

  // cannot determine spectral weights if no opacities
  if (opacities().empty() || opacities().find(op_name) == opacities().end()) {
    return {};
  }

  auto op = opacities().at(op_name);
  if (op.type().compare(0, 3, "rfm") == 0) {
    return read_dimvar_netcdf<double>(op.opacity_files()[0], "weights");
  } else if (op.type().compare(0, 9, "multiband") == 0) {
    return read_var_pt<double>(op.opacity_files()[0], "weights");
  } else {
    return {};
  }
}

RadiationBandImpl::RadiationBandImpl(RadiationBandOptions const& options_)
    : options(options_) {
  reset();

  // disort options maybe updated after initialization
  // reset it back
  if (options.solver_name() == "disort") {
    options.disort(rtsolver.get<disort::Disort>()->options);
  }
}

void RadiationBandImpl::reset() {
  auto str = options.outdirs();
  torch::Tensor ray_out;
  if (!str.empty()) {
    ray_out = parse_radiation_directions(str);
  }

  // create opacities
  for (auto const& [name, op] : options.opacities()) {
    if (op.type() == "jit") {
      opacities[name] = torch::nn::AnyModule(JITOpacity(op));
      nmax_prop_ = std::max((int)nmax_prop_, 2 + op.nmom());
    } else if (op.type() == "rfm-lbl") {
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
    } else if (op.type() == "multiband-ck") {
      auto a = MultiBand(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
      options.ww() = read_var_pt<double>(op.opacity_files()[0], "weights");
    } else if (op.type() == "wavetemp") {
      auto a = WaveTemp(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "fourcolumn") {
      auto a = FourColumn(op);
      nmax_prop_ = std::max((int)nmax_prop_, 2 + a->options.nmom());
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "helios") {
      auto a = Helios(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
      options.ww() = std::vector<double>(
          a->weights.data_ptr<double>(),
          a->weights.data_ptr<double>() + a->weights.numel());
    } else if (op.type() == "simple-grey") {
      auto a = SimpleGrey(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "freedman-mean") {
      auto a = FreedmanMean(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "jup-gas-vis") {
      auto a = JupGasVisible(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "jup-gas-ir") {
      auto a = JupGasIR(op);
      nmax_prop_ = std::max((int)nmax_prop_, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else {
      TORCH_CHECK(false, "Unknown attenuator type: ", op.type());
    }
    register_module(name, opacities[name].ptr());
  }

  // check waves are correctly set
  TORCH_CHECK(options.ww().size() > 0, "Spectral grid ww() undefined for band ",
              options.name());

  // create rtsolver
  auto [uphi, umu] = get_direction_grids<double>(ray_out);
  if (options.solver_name() == "jit") {
    // rtsolver = options.user().clone();
    register_module("solver", rtsolver.ptr());
  } else if (options.solver_name() == "disort") {
    rtsolver = torch::nn::AnyModule(disort::Disort(options.disort()));
    register_module("solver", rtsolver.ptr());
  } else {
    TORCH_CHECK(false, "Unknown solver: ", options.solver_name());
  }

  // create spectral grid
  ww = register_buffer("ww", torch::tensor(options.ww(), torch::kFloat64));
}

torch::Tensor RadiationBandImpl::forward(
    torch::Tensor conc, torch::Tensor dz,
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
  } else if (options.integration() == "weight") {
    if (options.solver_name() == "disort") {
      auto wmin = torch::tensor(options.disort().wave_lower(), ww.options());
      auto wmax = torch::tensor(options.disort().wave_upper(), ww.options());
      (*kwargs)["wavenumber"] = 0.5 * (wmin + wmax);
      (*kwargs)["wavelength"] = 1.e4 / (*kwargs)["wavenumber"];
    }
  }

  // bin optical properties
  auto prop =
      torch::zeros({ww.size(0), ncol, nlyr, nmax_prop_}, conc.options());

  for (auto& [_, a] : opacities) {
    auto kdata = a.forward(conc, *kwargs);
    int nprop = kdata.size(-1);

    // attenuation coefficients
    prop.select(-1, index::IEX) += kdata.select(-1, index::IEX);

    // attenuation weighted single scattering albedo
    if (nprop > 1) {
      prop.select(-1, index::ISS) +=
          kdata.select(-1, index::ISS) * kdata.select(-1, index::IEX);
    }

    // attenuation + single scattering albedo weighted phase moments
    if (nprop > 2) {
      prop.narrow(-1, index::IPM, nprop - 2) +=
          kdata.narrow(-1, index::IPM, nprop - 2) *
          (kdata.select(-1, index::ISS) * kdata.select(-1, index::IEX))
              .unsqueeze(-1);
    }
  }

  // average phase moments
  int nprop = prop.size(-1);
  if (nprop > 2) {
    prop.narrow(-1, index::IPM, nprop - 2) /=
        (prop.select(-1, index::ISS).unsqueeze(-1) + 1e-10);
  }

  // average single scattering albedo
  if (nprop > 1) {
    prop.select(-1, index::ISS) /= (prop.select(-1, index::IEX) + 1e-10);
  }

  // attenuation coefficients -> optical thickness
  prop.select(-1, index::IEX) *= dz.unsqueeze(0);

  // export band optical properties
  std::string op_name = "radiation/" + options.name() + "/opacity";
  shared[op_name] = prop;

  std::string spec_name = "radiation/" + options.name() + "/spectra";

  // run rt solver
  if (kwargs->find("temp") != kwargs->end()) {
    Layer2LevelOptions l2l;
    l2l.lower(kExtrapolate).upper(kExtrapolate).check_positivity(true);
    shared[spec_name] = rtsolver.forward(
        prop, bc, options.name(),
        std::make_optional(layer2level(dz, kwargs->at("temp"), l2l)));
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
