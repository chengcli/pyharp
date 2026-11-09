// harp
#include "calc_dz_hypsometric.hpp"

#include <harp/utils/layer2level.hpp>

namespace harp {

torch::Tensor calc_dz_hypsometric(torch::Tensor pres, torch::Tensor temp,
                                  torch::Tensor g_ov_R,
                                  std::optional<torch::Tensor> zf) {
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
    op.check_positivity(false);

    torch::Tensor lnp_levels;
    if (zf) {  // ln p linear in position between centres and past the ends
      TORCH_CHECK(nlyr > 1 && zf->size(-1) == nlyr + 1,
                  "zf must have nlyr + 1 faces and nlyr > 1");
      auto h = zf->diff(1, -1);
      auto hl = h.narrow(-1, 0, nlyr - 1), hr = h.narrow(-1, 1, nlyr - 1);
      auto inner = (hr * lnp.narrow(-1, 0, nlyr - 1) +
                    hl * lnp.narrow(-1, 1, nlyr - 1)) /
                   (hl + hr);
      auto l0 = lnp.narrow(-1, 0, 1), l1 = lnp.narrow(-1, 1, 1);
      auto u0 = lnp.narrow(-1, nlyr - 1, 1), u1 = lnp.narrow(-1, nlyr - 2, 1);
      auto lo = l0 + (l0 - l1) * h.narrow(-1, 0, 1) /
                         h.narrow(-1, 0, 2).sum(-1, true);
      auto hi = u0 + (u0 - u1) * h.narrow(-1, nlyr - 1, 1) /
                         h.narrow(-1, nlyr - 2, 2).sum(-1, true);
      lnp_levels = torch::cat({lo, inner, hi}, -1);
    } else {
      lnp_levels = layer2level(lnp, op);
    }
    dlnp = lnp_levels.slice(-1, 0, nlyr) - lnp_levels.slice(-1, 1, nlyr + 1);
  } else {
    TORCH_CHECK(false, "Invalid dimensions of pressure and temperature");
  }

  return temp * dlnp / g_ov_R;
}

}  // namespace harp
