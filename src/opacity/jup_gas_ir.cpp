// harp
#include "grey_opacities.hpp"

namespace harp {

torch::Tensor JupGasIRImpl::forward(
    torch::Tensor const &conc,
    std::map<std::string, torch::Tensor> const &kwargs) {
  Real p = var.w[IPR];
  Real T = var.w[IDN];

  Real dens = p / (pthermo->GetRd() * T);  // kg/m^3

  auto jstrat = 8.e-4 * p.pow(-0.5);  // IR opacity from hydrocarbons and haze
  auto cia = 2.e-8 * p;

  return options.scale() * dens * (cia + jstrat);  // -> 1/m
}

}  // namespace harp
