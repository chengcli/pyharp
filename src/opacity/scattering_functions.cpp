// harp
#include "scattering_functions.hpp"

namespace harp {

torch::Tensor henyey_greenstein(int nmom, torch::Tensor const& g) {
  TORCH_CHECK(torch::all((g > -1.) & (g < 1.)).item<bool>(),
              "henyey_greenstein::bad input variable g");
  auto vec = g.sizes().vec();
  vec.push_back(nmom);
  return torch::cumprod(g.unsqueeze(-1).expand(vec), -1);
}

torch::Tensor double_henyey_greenstein(int nmom, torch::Tensor const& ff,
                                       torch::Tensor const& g1,
                                       torch::Tensor const& g2) {
  auto result1 = henyey_greenstein(nmom, g1);
  auto result2 = henyey_greenstein(nmom, g2);

  return ff * result1 + (1.0 - ff) * result2;
}

}  // namespace harp
