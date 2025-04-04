// harp
#include "grey_opacities.hpp"

namespace harp {

torch::Tensor JupGasVisibleImpl::forward(
    torch::Tensor const &conc,
    std::map<std::string, torch::Tensor> const &kwargs) {
  Real p = var.w[IPR];
  Real T = var.w[IDN];

  auto dens = p / (pthermo->GetRd() * T);  // kg/m^3

  // this one is a good haze
  // Real result = 1.e-6*pow(p,0.5)+1.e-3*pow(p/1.e3, -2.); //visible opacity
  // with Jupiter haze Real strongch4 = 1.e-2*pow(p, -0.5); //visible opacity
  // with Jupiter haze
  auto strongch4 = 5.e-3 * pres.pow(-0.5);  // visible opacity with Jupiter haze
  double weakch4 = 0.;  // 1.e-3; //visible opacity with Jupiter haze
  // Real weakch4 = 1.e-3; //visible opacity with Jupiter haze

  // std::cout<<"scale=  " <<scale<<"  pres=  "<<p<< "  dens= "
  // <<dens<<std::endl;

  return options.scale() * dens * (strongch4 + weakch4);  // -> 1/m
}

}  // namespace harp
