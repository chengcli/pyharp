// harp
#include <harp/constants.h>

#include "grey_opacities.hpp"
#include "mean_molecular_weight.hpp"

namespace harp {

// coefficient from Richard S. Freedman 2014. APJS
const double FreedmanMeanImpl::c1 = 10.602;
const double FreedmanMeanImpl::c2 = 2.882;
const double FreedmanMeanImpl::c3 = 6.09e-15;
const double FreedmanMeanImpl::c4 = 2.954;
const double FreedmanMeanImpl::c5 = -2.526;
const double FreedmanMeanImpl::c6 = 0.843;
const double FreedmanMeanImpl::c7 = -5.490;
const double FreedmanMeanImpl::c13 = 0.8321;

torch::Tensor FreedmanMeanImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  auto c8 = torch::where(temp < 800., -14.051 * torch::ones_like(temp),
                         82.241 * torch::ones_like(temp));
  auto c9 = torch::where(temp < 800., 3.055 * torch::ones_like(temp),
                         -55.456 * torch::ones_like(temp));
  auto c10 = torch::where(temp < 800., 0.024 * torch::ones_like(temp),
                          8.754 * torch::ones_like(temp));
  auto c11 = torch::where(temp < 800., 1.877 * torch::ones_like(temp),
                          0.7048 * torch::ones_like(temp));
  auto c12 = torch::where(temp < 800., -0.445 * torch::ones_like(temp),
                          -0.0414 * torch::ones_like(temp));

  auto logp = torch::log10(pres * 10.);  // Pa to dyn/cm2
  auto logT = torch::log10(temp);

  logp.clamp_(0.);          // 1 microbar to 300 bar from Freedman
  logT.clamp_(log10(75.));  // 75 to 4000 K from Freedman

  auto klowp = c1 * torch::atan(logT - c2) -
               c3 / (logp + c4) * torch::exp(torch::pow(logT - c5, 2.0)) +
               c6 * options.metallicity() + c7;  // Eqn 4

  // Eqn 5
  auto khigp = c8 + c9 * logT + c10 * logT.pow(2.) + logp * (c11 + c12 * logT) +
               c13 * options.metallicity() *
                   (0.5 + 1. / M_PI * torch::atan((logT - 2.5) / 0.2));

  auto result = torch::pow(10.0, klowp) + torch::pow(10.0, khigp);  // cm^2/g

  auto mu = mean_molecular_weight(conc);
  auto dens = (pres * mu) / (constants::Rgas * temp);  // kg/m^3

  return options.scale() * 0.1 *
         (dens * result).unsqueeze(0).unsqueeze(-1);  // -> 1/m
}

}  // namespace harp
