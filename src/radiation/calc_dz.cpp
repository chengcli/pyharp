// harp
#include "calc_dz.hpp"

#include <utils/layer2level.hpp>

namespace harp {

torch::Tensor calc_dz(torch::Tensor pres, torch::Tensor temp,
                      torch::Tensor g_ov_R) {
  auto op = Layer2LevelOptions();
  op.order(k2ndOrder);
  op.lower(kExtrapolate);
  op.upper(kExtrapolate);

  int nlyr = pres.size(-1);
  auto new_lnp_levels = layer2level(pres.log(), op);

  auto dlnp =
      new_lnp_levels.slice(-1, 0, nlyr) - new_lnp_levels.slice(-1, 1, nlyr + 1);
  return dlnp / g_ov_R;
}

}  // namespace harp
