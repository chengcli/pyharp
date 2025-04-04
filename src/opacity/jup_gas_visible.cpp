// harp
#include <harp/constants.h>

#include "grey_opacities.hpp"
#include "mean_molecular_weight.hpp"

namespace harp {

torch::Tensor JupGasVisibleImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  auto mu = mean_molecular_weight(conc);

  auto dens = (pres * mu) / (constants::Rgas * temp);  // kg/m^3

  // this one is a good haze
  // Real result = 1.e-6*pow(p,0.5)+1.e-3*pow(p/1.e3, -2.); //visible opacity
  // with Jupiter haze Real strongch4 = 1.e-2*pow(p, -0.5); //visible opacity
  // with Jupiter haze
  auto strongch4 = 5.e-3 * pres.pow(-0.5);  // visible opacity with Jupiter haze
  double weakch4 = 0.;  // 1.e-3; //visible opacity with Jupiter haze
  // Real weakch4 = 1.e-3; //visible opacity with Jupiter haze

  // std::cout<<"scale=  " <<scale<<"  pres=  "<<p<< "  dens= "
  // <<dens<<std::endl;

  return (options.scale() * dens * (strongch4 + weakch4))
      .unsqueeze(0)
      .unsqueeze(-1);  // -> 1/m
}

}  // namespace harp
