// C/C++ headers
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

// harp
#include <utils/parse_radiation_direction.hpp>
#include <utils/vectorize.hpp>

#include "radiation.hpp"
// #include "rt_solvers.hpp"

namespace harp {
std::unordered_map<std::string, torch::Tensor> shared;

void RadiationOptions::set_flags(std::string const& str) {
  std::vector<std::string> dstr = Vectorize<std::string>(str.c_str(), " ,");

  for (int i = 0; i < dstr.size(); ++i) {
    if (dstr[i] == "time_dependent") {
      time_dependent(true);
    } else if (dstr[i] == "broad_band") {
      broad_band(true);
    } else if (dstr[i] == "stellar_beam") {
      stellar_beam(true);
    } else if (dstr[i] == "write_bin_radiance") {
      write_bin_radiance(true);
    } else {
      std::stringstream msg;
      msg << "flag: '" << dstr[i] << "' unrecognized" << std::endl;
      throw std::runtime_error("RadiationOptions::set_flags: " + msg.str());
    }
  }
}

RadiationImpl::RadiationImpl(RadiationOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationImpl::reset() {
  for (int i = 0; i < options.bands().size(); ++i) {
    auto name = options.bands()[i];
    // set default outgoing radiation directions
    if (!options.outdirs().empty()) {
      options.band_options()[i].outdirs(options.outdirs());
    }
    bands[name] = RadiationBand(options.band_options()[i]);
    register_module(name, bands[name]);
  }
}

torch::Tensor RadiationImpl::forward(torch::Tensor ftoa, torch::Tensor var_x,
                                     double ray[2]) {
  torch::Tensor out = torch::zeros_like(ftoa);

  torch::optional<torch::Tensor> area1 = torch::nullopt;
  torch::optional<torch::Tensor> vol = torch::nullopt;

  if (shared.find("coordinate/area1") != shared.end()) {
    area1 = shared["coordinate/area1"];
  }

  if (shared.find("coordinate/vol") != shared.end()) {
    vol = shared["coordinate/vol"];
  }

  /*if (options.flux_flag()) {
    for (auto& [name, band] : bands) {
      out += band->forward(x1f, ftoa, var_x, ray, area1, vol);
    }
  } else {
    for (auto& [name, band] : bands) {
      band->forward(x1f, ftoa, var_x, ray, area1, vol);
    }
  }*/

  return out;
}

}  // namespace harp
