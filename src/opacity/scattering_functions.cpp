// harp
#include "scattering_functions.hpp"

namespace harp {

torch::Tensor henyey_greenstein(int nmom, torch::Tensor const& g) {
  TORCH_CHECK(g.min().item<double>() > -1. && g.max().item<double>() < 1.,
              "henyey_greenstein::bad input variable g");
  auto vec = g.sizes().vec();
  vec.push_back(nmom);
  auto result = g.unsqueeze(-1).expand(vec).contiguous();

  for (int k = 0; k < nmom; k++) {
    result.select(-1, k).pow_(k + 1);
  }

  return result;
}

torch::Tensor double_henyey_greenstein(int nmom, torch::Tensor const& ff,
                                       torch::Tensor const& g1,
                                       torch::Tensor const& g2) {
  auto result1 = henyey_greenstein(nmom, g1);
  auto result2 = henyey_greenstein(nmom, g2);

  return ff * result1 + (1.0 - ff) * result2;
}

}  // namespace harp
