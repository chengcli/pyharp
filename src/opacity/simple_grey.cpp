// harp
#include <harp/constants.h>

#include "grey_opacities.hpp"
#include "mean_molecular_weight.hpp"

namespace harp {

torch::Tensor SimpleGreyImpl::forward(
    torch::Tensor const& conc,
    std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  auto mu = mean_molecular_weight(conc);
  auto dens = (pres * mu) / (Constants::Rgas * temp);  // kg/m^3

  // kappa = kappa_a * pow(p, kappa_b)
  auto kappa = options.kappa_a() * pres.pow(options.kappa_b());
  kappa
      .clamp_(options.kappa_cut())

      // Tan and Komacek 2019 simple fit m^2/kg
      //     k_VIS= 10.0_dp**(0.0478_dp*Pl10**2 - 0.1366_dp*Pl10 - 3.2095_dp)
      //     k_IR = 10.0_dp**(0.0498_dp*Pl10**2 - 0.1329_dp*Pl10 - 2.9457_dp)
      /*
        Real logp = log10(p); // Pa
      if (wave < 40000.) //for semigrey
        result = pow(10.0, (0.0498*pow(logp,2.) - 0.1329*logp - 2.9457));
      else
        result = pow(10.0, (0.0478*pow(logp,2.) - 0.1366*logp - 3.2095));
      */

      // Komacek et al. 2017
      // if (wave < 40000.) //for semigrey
      //   result = 2.28e-6*pow(p, 0.53);
      // else
      // result = 2.28e-6 * pow(p, 0.53);  // visible opacity scale in disort
      //

      // Shami intercomparison 2024 from Guillot
      // result = 1.e-3;

      return (dens * kappa)
      .unsqueeze(0)
      .unsqueeze(-1);  // -> 1/m
}
