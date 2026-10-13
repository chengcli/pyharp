// harp
#include "scattering_functions.hpp"

namespace harp {

torch::Tensor henyey_greenstein(int nmom, torch::Tensor const& g) {
  TORCH_CHECK(torch::all((g > -1.) & (g < 1.)).item<bool>(),
              "henyey_greenstein::bad input variable g");
  TORCH_CHECK(nmom >= 0, "henyey_greenstein::nmom must be nonnegative");

  if (nmom == 0) {
    return g.unsqueeze(-1).narrow(-1, 0, 0).clone();
  }

  std::vector<torch::Tensor> moments;
  moments.reserve(nmom);
  moments.push_back(g);
  for (int k = 1; k < nmom; ++k) {
    moments.push_back(moments.back() * g);
  }
  return torch::stack(moments, -1);
}

torch::Tensor double_henyey_greenstein(int nmom, torch::Tensor const& ff,
                                       torch::Tensor const& g1,
                                       torch::Tensor const& g2) {
  auto result1 = henyey_greenstein(nmom, g1);
  auto result2 = henyey_greenstein(nmom, g2);

  return ff * result1 + (1.0 - ff) * result2;
}

}  // namespace harp
