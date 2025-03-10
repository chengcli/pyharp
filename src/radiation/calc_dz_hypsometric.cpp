// harp
#include "calc_dz_hypsometric.hpp"

#include <utils/layer2level.hpp>

namespace harp {

torch::Tensor calc_dz_hypsometric(torch::Tensor pres, torch::Tensor temp,
                                  torch::Tensor g_ov_R) {
  int nlyr = temp.size(-1);
  auto lnp = pres.log();

  torch::Tensor dlnp;
  if (pres.size(-1) == nlyr + 1) {
    dlnp = lnp.slice(-1, 0, nlyr) - lnp.slice(-1, 1, nlyr + 1);
  } else if (pres.size(-1) == nlyr) {
    // pressure and temperature are at the layer centers
    auto op = Layer2LevelOptions();
    op.order(k2ndOrder);
    op.lower(kExtrapolate);
    op.upper(kExtrapolate);

    auto lnp_levels = layer2level(lnp, op);
    std::cout << "lnp_levels: " << lnp_levels << std::endl;
    dlnp = lnp_levels.slice(-1, 0, nlyr) - lnp_levels.slice(-1, 1, nlyr + 1);
  } else {
    TORCH_CHECK(false, "Invalid dimensions of pressure and temperature");
  }

  return temp * dlnp / g_ov_R;
}

}  // namespace harp
