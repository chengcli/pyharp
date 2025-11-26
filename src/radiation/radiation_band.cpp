// yaml
#include <yaml-cpp/yaml.h>

// disort
#include <disort/index.h>

// harp
#include <harp/opacity/fourcolumn.hpp>
#include <harp/opacity/helios.hpp>
#include <harp/opacity/jit_opacity.hpp>
#include <harp/opacity/multiband.hpp>
#include <harp/opacity/opacity_formatter.hpp>
#include <harp/opacity/rfm.hpp>
#include <harp/opacity/wavetemp.hpp>
#include <harp/utils/layer2level.hpp>
#include <harp/utils/parse_yaml_input.hpp>
#include <harp/utils/read_dimvar_netcdf.hpp>
#include <harp/utils/read_var_pt.hpp>
#include <harp/utils/strings.hpp>

#include "disort_config.hpp"
#include "flux_utils.hpp"
#include "get_direction_grids.hpp"
#include "parse_radiation_direction.hpp"
#include "radiation.hpp"
#include "radiation_band.hpp"
#include "radiation_formatter.hpp"

namespace harp {

RadiationBandOptions RadiationBandOptionsImpl::from_yaml(
    std::string const& filename, std::string const& bd_name) {
  auto config = YAML::LoadFile(filename);
  auto op = std::make_shared<RadiationBandOptionsImpl>();

  // band configuration
  TORCH_CHECK(config["bands"], "'bands' not found in ", filename);

  YAML::Node band;
  bool found = false;
  for (auto const& bd : config["bands"]) {
    auto name = bd["name"].as<std::string>();
    if (name == bd_name) {
      band = static_cast<YAML::Node>(bd);
      found = true;
      break;
    }
  }

  TORCH_CHECK(found, "band ", bd_name, " not found in ", filename);
  op->verbose(band["verbose"].as<bool>(false));

  TORCH_CHECK(band["opacities"], "opacities not found in band ", bd_name);

  for (auto const& opa : band["opacities"]) {
    std::string op_name = opa.as<std::string>();
    if (op->verbose()) {
      std::cout << "  Loading opacity '" << op_name << "' for band '" << bd_name
                << "'..." << std::endl;
    }
    op->opacities()[op_name] =
        OpacityOptionsImpl::from_yaml(filename, op_name, bd_name);
  }

  auto [wmin, wmax] = parse_wave_range(band);
  if (op->verbose()) {
    std::cout << "  Wavenumber range: " << wmin << " - " << wmax << " cm-1"
              << std::endl;
  }

  // number of radiation waves
  TORCH_CHECK(band["nwave"], "'nwave' not found in band ", bd_name);
  op->nwave(band["nwave"].as<int>());
  if (op->verbose()) {
    std::cout << "  Number of waves: " << op->nwave() << std::endl;
  }

  // number of radiation streams (default 4)
  op->nstr(band["nstr"].as<int>(4));
  if (op->verbose()) {
    std::cout << "  Number of streams: " << op->nstr() << std::endl;
  }

  // number of columns and layers
  TORCH_CHECK(config["geometry"], "'geometry' not found in ", filename);
  TORCH_CHECK(config["geometry"]["cells"], "'cells' not found in geometry");
  auto cells = config["geometry"]["cells"];
  TORCH_CHECK(cells["nx1"], "'nx1' not found in cells");

  op->nlyr(cells["nx1"].as<int>());
  op->ncol(cells["nx2"].as<int>(1) * cells["nx3"].as<int>(1));

  op->name(bd_name);

  TORCH_CHECK(band["solver"], "'solver' not found in band ", bd_name);

  op->solver_name(band["solver"].as<std::string>());
  if (op->solver_name() == "disort") {
    op->disort() =
        create_disort_config(op->nwave(), op->ncol(), op->nlyr(), op->nstr());
    op->disort()->header("running disort " + bd_name);
    op->disort()->wave_lower(std::vector<double>(op->nwave(), wmin));
    op->disort()->wave_upper(std::vector<double>(op->nwave(), wmax));
    if (band["flags"]) {
      op->disort()->flags(trim_copy(band["flags"].as<std::string>()));
    }
    if (op->verbose()) {
      std::cout << "  Solver flags: " << op->disort()->flags() << std::endl;
    }
  } else if (op->solver_name() == "twostr") {
    TORCH_CHECK(false, "twostr solver not implemented");
  } else {
    TORCH_CHECK(false, "unknown solver: ", op->solver_name());
  }

  return op;
}

RadiationBandImpl::RadiationBandImpl(RadiationBandOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationBandImpl::reset() {
  auto str = options->outdirs();
  torch::Tensor ray_out;
  if (!str.empty()) {
    ray_out = parse_radiation_directions(str);
  }

  // create opacities
  int nmax_prop = 1;

  for (auto const& [name, op] : options->opacities()) {
    if (op->type() == "jit") {
      opacities[name] = torch::nn::AnyModule(JITOpacity(op));
      nmax_prop = std::max(nmax_prop, 2 + op->nmom());
    } else if (op->type() == "rfm-lbl") {
      auto a = RFM(op);
      nmax_prop = std::max(nmax_prop, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op->type() == "rfm-ck") {
      auto a = RFM(op);
      nmax_prop = std::max(nmax_prop, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op->type() == "multiband-ck") {
      auto a = MultiBand(op);
      nmax_prop = std::max(nmax_prop, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op->type() == "wavetemp") {
      auto a = WaveTemp(op);
      nmax_prop = std::max(nmax_prop, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op->type() == "fourcolumn") {
      auto a = FourColumn(op);
      nmax_prop = std::max(nmax_prop, 2 + a->options->nmom());
      opacities[name] = torch::nn::AnyModule(a);
    } else if (op->type() == "helios") {
      auto a = Helios(op);
      nmax_prop = std::max(nmax_prop, 1);
      opacities[name] = torch::nn::AnyModule(a);
    } else {
      TORCH_CHECK(false, "Unknown opacity type: ", op->type());
    }
    register_module(name, opacities[name].ptr());
  }

  // create rtsolver
  auto [uphi, umu] = get_direction_grids<double>(ray_out);
  if (options->solver_name() == "jit") {
    // rtsolver = options->user().clone();
    register_module("solver", rtsolver.ptr());
  } else if (options->solver_name() == "disort") {
    rtsolver = torch::nn::AnyModule(disort::Disort(options->disort()));
    register_module("solver", rtsolver.ptr());
  } else {
    TORCH_CHECK(false, "Unknown solver: ", options->solver_name());
  }

  // check either wavenumber or weight is set
  TORCH_CHECK(options->wavenumber().size() > 0 || options->weight().size() > 0,
              "Either 'wavenumber' or 'weight' must be set in band ",
              options->name());

  // create optical properties holder
  prop =
      register_buffer("prop", torch::zeros({options->nwave(), options->ncol(),
                                            options->nlyr(), nmax_prop},
                                           torch::kFloat64));

  // create bin spectrum holder
  spectrum = register_buffer("spectrum", torch::tensor({0.}, torch::kFloat64));
}

torch::Tensor RadiationBandImpl::forward(
    torch::Tensor conc, torch::Tensor dz,
    std::map<std::string, torch::Tensor>* bc,
    std::map<std::string, torch::Tensor>* kwargs) {
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  // check if dimensions are consistent
  if (dz.dim() > 1) {
    TORCH_CHECK(dz.size(0) == ncol, "'dz' size(", dz.size(0),
                ") inconsistent with 'conc' ncol(", ncol, ")");
  }
  TORCH_CHECK(dz.size(-1) == nlyr, "'dz' size(", dz.size(-1),
              ") inconsistent with 'conc' nlyr(", nlyr, ")");
  TORCH_CHECK(options->ncol() == ncol, "'conc' ncol(", ncol,
              ") inconsistent with options ncol(", options->ncol(), ")");
  TORCH_CHECK(options->nlyr() == nlyr, "'conc' nlyr(", nlyr,
              ") inconsistent with options nlyr(", options->nlyr(), ")");

  // add wavelength or wavenumber to kwargs, overwrite existing values
  int nwave = 0;
  if (options->weight().size() > 0) {
    (*kwargs)["wavenumber"] =
        torch::tensor(options->wavenumber(), conc.options());
    (*kwargs)["wavelength"] = 1.e4 / (*kwargs)["wavenumber"];
    nwave = options->wavenumber().size();
  } else {
    nwave = options->weight().size();
  }
  if (options->verbose()) {
    std::cout << "  Calculating " << nwave << " waves..." << std::endl;
  }

  // check nwave consistency
  TORCH_CHECK(nwave == options->nwave(),
              "'nwave' inconsistent with options nwave(", options->nwave(),
              "), got ", nwave);

  prop.zero_();

  for (auto& [aname, a] : opacities) {
    auto kdata = a.forward(conc, *kwargs);
    int nprop = kdata.size(-1);

    // attenuation coefficients
    prop.select(-1, disort::IEX) += kdata.select(-1, disort::IEX);

    // attenuation weighted single scattering albedo
    if (nprop > 1) {
      prop.select(-1, disort::ISS) +=
          kdata.select(-1, disort::ISS) * kdata.select(-1, disort::IEX);
    }

    // attenuation + single scattering albedo weighted phase moments
    if (nprop > 2) {
      prop.narrow(-1, disort::IPM, nprop - 2) +=
          kdata.narrow(-1, disort::IPM, nprop - 2) *
          (kdata.select(-1, disort::ISS) * kdata.select(-1, disort::IEX))
              .unsqueeze(-1);
    }

    if (options->verbose()) {
      std::cout << "  Calculating opacity contribution from: " << aname
                << std::endl;
    }
  }

  // average phase moments
  int nprop = prop.size(-1);
  if (nprop > 2) {
    prop.narrow(-1, disort::IPM, nprop - 2) /=
        (prop.select(-1, disort::ISS).unsqueeze(-1) + 1e-10);
    if (options->verbose()) {
      std::cout << "  Averaging phase moments." << std::endl;
    }
  }

  // average single scattering albedo
  if (nprop > 1) {
    prop.select(-1, disort::ISS) /= (prop.select(-1, disort::IEX) + 1e-10);
    if (options->verbose()) {
      std::cout << "  Averaging single scattering albedo." << std::endl;
    }
  }

  // attenuation coefficients -> optical thickness
  prop.select(-1, disort::IEX) *= dz.unsqueeze(0);
  if (options->verbose()) {
    std::cout << "  Converting attenuation coefficients to optical thickness."
              << std::endl;
  }

  // run rt solver
  if (kwargs->find("tempf") != kwargs->end()) {
    int nlyr = prop.size(-1);
    int nlev = kwargs->at("tempf").size(-1);
    TORCH_CHECK(nlev == nlyr + 1, "'tempf' size must be nlyr + 1 = ", nlyr + 1,
                ", got ", nlev);
    // positivity check
    if (torch::any(kwargs->at("tempf") < 0).item<bool>()) {
      TORCH_CHECK(false, "Negative values found in 'tempf'");
    }

    spectrum.set_(rtsolver.forward(prop, bc, options->name(),
                                   std::make_optional(kwargs->at("tempf"))));
    if (options->verbose()) {
      std::cout << "  Done running rt solver with level temperatures."
                << std::endl;
    }
  } else if (kwargs->find("temp") != kwargs->end()) {
    Layer2LevelOptions l2l;
    l2l.order(options->l2l_order());
    l2l.lower(kExtrapolate).upper(kExtrapolate).check_positivity(true);
    spectrum.set_(rtsolver.forward(
        prop, bc, options->name(),
        std::make_optional(layer2level(dz, kwargs->at("temp"), l2l))));

    if (options->verbose()) {
      std::cout
          << "  Done running rt solver with interpolated layer temperatures."
          << std::endl;
    }
  } else {
    if (options->verbose()) {
      std::cout << "  No temperatures provided" << std::endl;
    }

    spectrum.set_(rtsolver.forward(prop, bc, options->name()));

    if (options->verbose()) {
      std::cout << "  Done running rt solver without temperatures."
                << std::endl;
    }
  }

  // accumulate flux from flux spectrum
  torch::Tensor result;
  if (options->weight().size() > 0) {
    result = sum_spectrum(spectrum,
                          torch::tensor(options->weight(), spectrum.options()),
                          "weight");
  } else {
    result = sum_spectrum(
        spectrum, torch::tensor(options->wavenumber(), spectrum.options()),
        "wavenumber");
  }
  return result;
}

void RadiationBandImpl::pretty_print(std::ostream& out) const {
  out << "RadiationBand: " << options->name() << std::endl;
  out << "Absorbers: (";
  for (auto const& [name, _] : opacities) {
    out << name << ", ";
  }
  out << ")" << std::endl;
  out << std::endl << "Solver: " << options->solver_name();
}

}  // namespace harp
