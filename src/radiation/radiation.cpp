// harp
#include "radiation.hpp"

#include "flux_utils.hpp"

namespace harp {

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
    fluxes[name] = band->forward(conc, path, bc, kwargs);
    if (first_band) {
      total_flux = fluxes[name].clone();
      first_band = false;
    } else {
      total_flux += fluxes[name];
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

  downward = cal_surface_flux(total_flux);
  upward = cal_toa_flux(total_flux);

  return net_flux;
}

}  // namespace harp
