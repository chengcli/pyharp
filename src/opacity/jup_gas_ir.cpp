// harp
#include <harp/constants.h>

#include "grey_opacities.hpp"
#include "mean_molecular_weight.hpp"

namespace harp {

torch::Tensor JupGasIRImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  auto mu = mean_molecular_weight(conc);
  auto dens = (pres * mu) / (constants::Rgas * temp);  // kg/m^3

  auto jstrat =
      8.e-4 * pres.pow(-0.5);  // IR opacity from hydrocarbons and haze
  auto cia = 2.e-8 * pres;

  return (options.scale() * dens * (cia + jstrat))
      .unsqueeze(0)
      .unsqueeze(-1);  // -> 1/m
}

}  // namespace harp
