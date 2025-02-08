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

#include "radiation.hpp"
#include "radiation_band.hpp"

namespace harp {
RadiationBandImpl::RadiationBandImpl(RadiationBandOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationBandImpl::reset() {
  TORCH_CHECK(options.wave_lower().size() == options.wave_upper().size(),
              "wave_lower and wave_upper must have the same size");
  int nwave = options.wave_lower().size();

  auto str = options.outdirs();
  torch::Tensor ray_out;
  if (!str.empty()) {
    ray_out = parse_radiation_directions(str);
  }

  // create attenuators
  for (auto const& [name, op] : options.attenuators()) {
    if (op.type() == "rfm") {
      auto a = RFM(op);
      options.nmax_prop() = std::max(options.nmax_prop(), a->kdata.size(1));
      attenuators[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "s8_fuller") {
      auto a = S8Fuller(op);
      options.nmax_prop() = std::max(options.nmax_prop(), a->kdata.size(1));
      attenuators[name] = torch::nn::AnyModule(a);
    } else if (op.type() == "h2sO4_simple") {
      auto a = H2SO4Simple(op);
      options.nmax_prop() = std::max(options.nmax_prop(), a->kdata.size(1));
      attenuators[name] = torch::nn::AnyModule(a);
    } else {
      TORCH_CHECK(false, "Unknown attenuator type: ", op.type());
    }
    register_module(name, attenuators[name].ptr());
  }

  // create rtsolver
  auto [uphi, umu] = get_direction_grids<double>(ray_out);
  if (options.solver_name() == "disort") {
    options.disort().ds().nlyr = options.nlyr();

    options.disort().nwave(nwave);
    options.disort().ncol(options.ncol());

    options.disort().user_phi(uphi);
    options.disort().user_mu(umu);
    options.disort().wave_lower(options.wave_lower());
    options.disort().wave_upper(options.wave_upper());

    rtsolver = torch::nn::AnyModule(disort::Disort(options.disort()));
    register_module("solver", rtsolver.ptr());
  } else {
    TORCH_CHECK(false, "Unknown solver: ", options.solver_name());
  }
}

torch::Tensor RadiationBandImpl::forward(
    torch::Tensor conc, torch::Tensor dz,
    std::map<std::string, torch::Tensor>& bc,
    std::map<std::string, torch::Tensor> const& op) {
  // bin optical properties
  TORCH_CHECK(conc.size(0) == options.ncol());
  TORCH_CHECK(conc.size(1) == options.nlyr());

  auto prop = torch::zeros(
      {options.nmax_prop(), options.ncol(), options.nlyr()}, conc.options());

  for (auto& [_, a] : attenuators) {
    auto kdata = a.forward(conc, op);
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
  if (op.count("temp") > 0) {
    return rtsolver.forward(prop, &bc,
                            layer2level(op.at("temp"), options.l2l()));
  } else {
    return rtsolver.forward(prop, &bc);
  }

  /* accumulate flux from spectral bins
  auto bflx = (flx * weight.view({-1, 1, 1, 1})).sum(0);

  if (coord.count("area") == 0) {
    return bflx;
  } else {
    return spherical_flux_correction(bflx, x1f, coord.at("area"),
  coord.at("vol"));
  }*/
}

void RadiationBandImpl::pretty_print(std::ostream& out) const {
  out << "RadiationBand: " << options.name() << std::endl;
  out << "Absorbers: (";
  for (auto const& [name, _] : attenuators) {
    out << name << ", ";
  }
  out << ")" << std::endl;
  out << std::endl << "Solver: " << options.solver_name();
}

}  // namespace harp
