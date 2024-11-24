// C/C++
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// harp
#include <index.h>

#include <opacity/attenuator.hpp>
#include <registry.hpp>
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
  spec = register_buffer("spec",
                         torch::tensor({options.nspec(), 2}, torch::kFloat32));

  prop =
      register_buffer("prop", torch::zeros({3 + options.nstr(), options.nc3(),
                                            options.nc2(), options.nc1()},
                                           torch::kFloat32));

  auto str = options.outdirs();
  if (!str.empty()) {
    rayOutput = register_buffer("rayOutput", parse_radiation_directions(str));
  }

  // create attenuators
  for (int i = 0; i < options.attenuators().size(); ++i) {
    auto name = options.attenuators()[i];
    attenuators[name] =
        register_module_op(this, name, options.attenuator_options()[i]);
  }

  // create rt solver
  if (options.solver_name() == "Disort") {
    options.disort_options().ds().wvnmlo = options.wmin();
    options.disort_options().ds().wvnmhi = options.wmax();
    options.disort_options().ds().nstr = options.nstr();
    options.disort_options().ds().nphase = options.nstr();
    options.disort_options().ds().nmom = options.nstr();
    options.disort_options().ds().nlyr = options.nc1();

    auto [uphi, umu] = get_direction_grids<double>(rayOutput);
    options.disort_options().ds().nphi = std::max(1uL, uphi.size());
    options.disort_options().ds().numu = std::max(1uL, umu.size());

    solver = register_module_op(this, "solver", options.disort_options());

    for (int i = 0; i < umu.size(); ++i) {
      options.disort_options().ds().umu[i] = umu[i];
    }

    for (int i = 0; i < uphi.size(); ++i) {
      options.disort_options().ds().phi[i] = uphi[i];
    }
  }
}

torch::Tensor RadiationBandImpl::forward(torch::Tensor x1f, torch::Tensor ftoa,
                                         torch::Tensor var_x, float ray[2],
                                         torch::optional<torch::Tensor> area,
                                         torch::optional<torch::Tensor> vol) {
  prop.fill_(0.);

  for (auto& [_, a] : attenuators) {
    auto kdata = a->forward(var_x);
    prop[IAB] += kdata[IAB];
    prop[ISS] += kdata[ISS] * kdata[IAB];
    prop.slice(0, IPM, -1) += kdata.slice(0, IPM, -1) * kdata[ISS] * kdata[IAB];
  }

  // absorption coefficients -> optical thickness
  prop.slice(0, IPM, -1) /= (prop[ISS] + 1e-10);
  prop[ISS] /= (prop[IAB] + 1e-10);
  prop[IAB] *= x1f.slice(0, 1, -1) - x1f.slice(0, 0, -2);

  // export aggregated band properties
  std::string name = "radiation/" + options.name() + "/optics";
  shared[name] =
      std::async(std::launch::async, [&]() {
        return torch::sum(prop * spec[IWT].view({1, -1, 1, 1, 1}), 1);
      }).share();

  auto temf = layer2level(var_x[ITM], options.l2l_options());

  if (options.solver_name() == "Disort") {
    // mu
    options.disort_options().ds().bc.umu0 = ray[0] > 1.E-3 ? ray[0] : 1.E-3;

    // phi
    options.disort_options().ds().bc.phi0 = ray[1];
  }

  auto flx = solver->forward(prop, ftoa, temf);

  /// accumulate flux from spectral bins
  auto bflx = torch::sum(flx * spec[IWT].view({1, -1, 1, 1, 1}), 0);

  if (!area.has_value() || !vol.has_value()) {
    return bflx;
  } else {
    return spherical_flux_correction(bflx, x1f, area.value(), vol.value());
  }
}

std::string RadiationBandImpl::to_string() const {
  std::stringstream ss;
  ss << "RadiationBand: " << options.name() << std::endl;
  ss << "Absorbers: [";
  for (auto const& [name, _] : attenuators) {
    ss << name << ", ";
  }
  ss << "]" << std::endl;
  // ss << std::endl << "RT-Solver: " << psolver_->GetName();
  return ss.str();
}

/*std::shared_ptr<RadiationBand::RTSolver> RadiationBand::CreateRTSolverFrom(
    std::string const &rt_name, YAML::Node const &rad) {
  std::shared_ptr<RTSolver> psolver;

  if (rt_name == "Lambert") {
    psolver = std::make_shared<RTSolverLambert>(this, rad);
#ifdef RT_DISORT
  } else if (rt_name == "Disort") {
    psolver = std::make_shared<RTSolverDisort>(this, rad);
#endif  // RT_DISORT
  } else {
    throw NotFoundError("RadiationBand", rt_name);
  }

  return psolver;
}*/

}  // namespace harp
