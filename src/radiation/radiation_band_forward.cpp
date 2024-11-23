// harp
#include <index.h>

#include "radiation.hpp"
#include "radiation_band.hpp"

namespace harp {
torch::Tensor RadiationBandImpl::forward(torch::Tensor x1f, torch::Tensor ftoa,
                                         torch::Tensor var_x) {
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

  auto bflx = torch::zeros(
      {options.nspec(), 2, options.nc3(), options.nc2(), options.nc1()},
      var_x.options());

  auto temf = layer2level(var_x[ITM], options.l2l_options());

  // solver->forward(ftoa, temf, prop, bflx);
  return torch::sum(bflx * spec[IWT].view({-1, 1, 1, 1, 1}), 0);
}
}  // namespace harp
