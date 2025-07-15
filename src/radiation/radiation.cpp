// yaml
#include <yaml-cpp/yaml.h>

// harp
#include <harp/compound.hpp>

#include "flux_utils.hpp"
#include "radiation.hpp"

namespace harp {

std::unordered_map<std::string, torch::Tensor> shared;
std::vector<std::string> species_names;
std::vector<double> species_weights;

RadiationOptions RadiationOptions::from_yaml(std::string const& filename) {
  RadiationOptions rad;
  auto config = YAML::LoadFile(filename);

  // null-op
  if (!config["bands"]) return rad;

  // check if species are defined
  TORCH_CHECK(config["species"],
              "'species' is not defined in the radiation configuration file");

  species_names.clear();
  species_weights.clear();

  for (const auto& sp : config["species"]) {
    species_names.push_back(sp["name"].as<std::string>());
    std::map<std::string, double> comp;

    for (const auto& it : sp["composition"]) {
      std::string key = it.first.as<std::string>();
      double value = it.second.as<double>();
      comp[key] = value;
    }
    species_weights.push_back(get_compound_weight(comp));
  }

  for (auto bd : config["bands"]) {
    auto bd_name = bd.as<std::string>();
    rad.bands()[bd_name] = RadiationBandOptions::from_yaml(bd_name, config);
  }

  return rad;
}

RadiationImpl::RadiationImpl(RadiationOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationImpl::reset() {
  for (auto& [name, bop] : options.bands()) {
    // set default outgoing radiation directions
    if (!options.outdirs().empty()) {
      bop.outdirs(options.outdirs());
    }
    bands[name] = RadiationBand(bop);
    register_module(name, bands[name]);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RadiationImpl::forward(
    torch::Tensor conc, torch::Tensor dz,
    std::map<std::string, torch::Tensor>* bc,
    std::map<std::string, torch::Tensor>* kwargs) {
  torch::Tensor total_flux;
  bool first_band = true;

  for (auto& [name, band] : bands) {
    std::string name1 = "radiation/" + name + "/total_flux";
    shared[name1] = band->forward(conc, dz, bc, kwargs);
    if (first_band) {
      total_flux = shared[name1].clone();
      first_band = false;
    } else {
      total_flux += shared[name1];
    }
  }

  auto net_flux = cal_net_flux(total_flux);

  // doing spherical scaling
  if (kwargs->find("area") != kwargs->end()) {
    auto kappa = spherical_flux_scaling(net_flux, kwargs->at("dz"),
                                        kwargs->at("area"), kwargs->at("vol"));
    total_flux *= kappa.unsqueeze(-1);
    net_flux *= kappa;
  }

  return std::make_tuple(net_flux, cal_surface_flux(total_flux),
                         cal_toa_flux(total_flux));
}

}  // namespace harp
