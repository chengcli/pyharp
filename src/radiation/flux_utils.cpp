// harp
#include "flux_utils.hpp"

#include <index.h>

#include <math/trapezoid.hpp>

namespace harp {

torch::Tensor cal_total_flux(torch::Tensor flux, torch::Tensor wave_or_weight,
                             std::string input) {
  // Check 1D tensor
  TORCH_CHECK(wave_or_weight.dim() == 1, "wave_or_weight must be 1D tensor");

  if (input == "wave") {
    return trapezoid(flux, wave_or_weight, /*dim=*/0);
  } else if (input == "weight") {
    return (flux * wave_or_weight.unsqueeze(1).expand_as(flux)).sum(0);
  } else {
    TORCH_CHECK(false, "input must be either 'wave' or 'weight'");
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

}  // namespace harp
