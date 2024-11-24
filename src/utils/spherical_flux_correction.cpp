#include "spherical_flux_correction.hpp"

torch::Tensor spherical_flux_correction(torch::Tensor flx, torch::Tensor x1f,
                                        torch::Tensor area, torch::Tensor vol) {
  auto nx1 = x1f.size(0);
  auto flx_iu = flx.select(-1, nx1 - 1);

  for (int i = nx1 - 2; i >= 0; --i) {
    auto dx1f = x1f[i + 1] - x1f[i];
    auto volh = (flx_iu - flx.select(-1, i)) / dx1f * vol.select(-1, i);
    flx_iu = flx.select(-1, i);
    flx.select(-1, i) =
        (flx.select(-1, i + 1) * area.select(-1, i + 1) - volh) /
        area.select(-1, i);
  }

  return flx;
}
