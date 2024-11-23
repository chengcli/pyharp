// C/C++
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

// opacity

// harp
#include <opacity/attenuator.hpp>
#include <registry.hpp>
#include <utils/parse_radiation_direction.hpp>

#include "radiation.hpp"
#include "radiation_band.hpp"
// #include "rt_solvers.hpp"

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
  // rt_solver = CreateRTSolverFrom(my["rt-solver"].as<std::string>(), rad);
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
