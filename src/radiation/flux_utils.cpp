// harp
#include "flux_utils.hpp"

#include <harp/index.h>

#include <harp/math/trapezoid.hpp>

namespace harp {

torch::Tensor cal_total_flux(torch::Tensor flux, torch::Tensor wave_or_weight,
                             std::string input) {
  // Check 1D tensor
  TORCH_CHECK(wave_or_weight.dim() == 1, "wave_or_weight must be 1D tensor");

  if (input == "wavelength" || input == "wavenumber") {
    return trapezoid(flux, wave_or_weight, /*dim=*/0);
  } else if (input == "weight") {
    int nwave = wave_or_weight.size(0);
    return (flux * wave_or_weight.view({nwave, 1, 1, 1})).sum(0);
  } else {
    TORCH_CHECK(false,
                "input must be either 'wavelength', 'wavenumber', or 'weight'");
  }
}

torch::Tensor cal_net_flux(torch::Tensor flux) {
  // Check last dimension
  TORCH_CHECK(flux.size(-1) == 2, "flux must have last dimension of size 2");
  return flux.select(-1, index::IUP) - flux.select(-1, index::IDN);
}

torch::Tensor cal_surface_flux(torch::Tensor flux) {
  // Check last dimension
  TORCH_CHECK(flux.size(-1) == 2, "flux must have last dimension of size 2");
  return flux.select(-1, index::IDN).select(-1, 0);
}

torch::Tensor cal_toa_flux(torch::Tensor flux) {
  // Check last dimension
  TORCH_CHECK(flux.size(-1) == 2, "flux must have last dimension of size 2");
  return flux.select(-1, index::IUP).select(-1, -1);
}

torch::Tensor spherical_flux_scaling(torch::Tensor flx, torch::Tensor dz,
                                     torch::Tensor area, torch::Tensor vol) {
  int nx1 = dz.size(-1);
  auto kappa = torch::ones_like(flx);
  auto volh = (flx.narrow(-1, 1, nx1) - flx.narrow(-1, 0, nx1)) / dz * vol;

  for (int i = nx1 - 1; i >= 0; --i) {
    kappa.select(-1, i) = (kappa.select(-1, i + 1) * flx.select(-1, i + 1) *
                               area.select(-1, i + 1) -
                           volh.select(-1, i)) /
                          area.select(-1, i);
  }

  return kappa;
}

}  // namespace harp
