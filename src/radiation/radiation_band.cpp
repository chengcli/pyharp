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
  TORCH_CHECK(options.wave_lower().size() == options.wave_upper().size(),
              "wave_lower and wave_upper must have the same size");
  int nwave = options.wave_lower().size();

  wave = register_buffer(
      "wave", (torch::tensor(options.wave_lower(), torch::kFloat64) +
               torch::tensor(options.wave_upper(), torch::kFloat64)) /
                  2.0);

  weight = register_buffer("weight", torch::ones({nwave}, torch::kFloat64));
  weight /= nwave;

  prop = register_buffer(
      "prop", torch::zeros({NMAX_RT_PROP, options.ncol(), options.nlyr()},
                           torch::kFloat64));

  auto str = options.outdirs();
  if (!str.empty()) {
    rayOutput = register_buffer("rayOutput", parse_radiation_directions(str));
  }

  // create attenuators
  for (auto const& [name, op] : options.attenuators()) {
    if (op.type() == "RFM") {
      attenuators[name] = torch::nn::AnyModule(AbsorberRFM(op));
    } else if (op.type() == "S8Fuller") {
      attenuators[name] = torch::nn::AnyModule(S8Fuller(op));
    } else if (op.type() == "H2SO4Simple") {
      attenuators[name] = torch::nn::AnyModule(H2SO4Simple(op));
    } else {
      TORCH_CHECK(false, "Unknown attenuator type: ", op.type());
    }
    register_module(name, attenuator[name].ptr());
  }

  // create rtsolver
  auto [uphi, umu] = get_direction_grids<double>(rayOutput);
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
    torch::Tensor x1f, torch::Tensor conc,
    std::map<std::string, torch::Tensor>& bc,
    torch::optional<torch::Tensor> pres, torch::optional<torch::Tensor> temp,
    torch::optional<torch::Tensor> area, torch::optional<torch::Tensor> vol) {
  prop.fill_(0.);
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  TORCH_CHECK(ncol == options.ncol(), "conc.size(0) != ncol");
  TORCH_CHECK(nlyr == options.nlyr(), "conc.size(1) != nlyr");
  TORCH_CHECK(x1f.size(0) == nlyr + 1, "x1f.size(0) != nlyr + 1");

  for (auto& [_, a] : attenuators) {
    auto kdata = a->forward(wave, conc);
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
  if (NMAX_RT_PROP > 2) {
    prop.narrow(0, index::IPM, NMAX_RT_PROP - 2) /= (prop[index::ISS] + 1e-10);
  }

  if (NMAX_RT_PROP > 1) {
    prop[index::ISS] /= (prop[index::IEX] + 1e-10);
  }

  prop[index::IEX] *= x1f.narrow(0, 1, nlyr) - x1f.narrow(0, 0, nlyr);

  // export aggregated band properties
  std::string name = "radiation/" + options.name() + "/optics";
  shared[name] = (prop * weight.view({-1, 1, 1, 1})).sum(0);

  torch::Tensor temf;
  if (temp.has_value()) {
    TORCH_CHECK(temp.value().size(0) == ncol,
                "DisortImpl::forward: temp.size(0) != ncol");
    TORCH_CHECK(temp.value().size(1) == nlyr + 1,
                "DisortImpl::forward: temp.size(1) != nlyr + 1");
    temf = layer2level(temp.value(), options.l2l());
  } else {
    temf = torch::zeros({1, 1}, prop.options()).expand({ncol, nlyr + 1});
  }

  auto flx = rtsolver->forward(prop, bc);

  // accumulate flux from spectral bins
  auto bflx = (flx * weight.view({-1, 1, 1, 1})).sum(0);

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
