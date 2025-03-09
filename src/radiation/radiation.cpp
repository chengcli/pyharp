// harp
#include "radiation.hpp"

#include "flux_utils.hpp"

namespace harp {

std::unordered_map<std::string, torch::Tensor> shared;

RadiationImpl::RadiationImpl(RadiationOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationImpl::reset() {
  for (auto& [name, bop] : options.band_options()) {
    // set default outgoing radiation directions
    if (!options.outdirs().empty()) {
      bop.outdirs(options.outdirs());
    }
    bands[name] = RadiationBand(bop);
    register_module(name, bands[name]);
  }
}

torch::Tensor RadiationImpl::forward(
    torch::Tensor conc, torch::Tensor path,
    std::map<std::string, torch::Tensor>* bc,
    std::map<std::string, torch::Tensor>* kwargs) {
  torch::Tensor total_flux;
  bool first_band = true;

  for (auto& [name, band] : bands) {
    std::string name1 = "radiation/" + name + "/total_flux";
    shared[name1] = band->forward(conc, path, bc, kwargs);
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

  // export downward flux to surface
  shared["radiation/downward_flux"] = cal_surface_flux(total_flux);

  // export upward flux out of the top of atmosphere
  shared["radiation/upward_flux"] = cal_toa_flux(total_flux);

  return net_flux;
}

}  // namespace harp
