#pragma once

// C/C++
#include <vector>

// torch
#include <torch/torch.h>

namespace harp {

extern std::vector<double> species_weights;

inline torch::Tensor mean_molecular_weight(torch::Tensor conc) {
  TORCH_CHECK(species_weights.size() == static_cast<size_t>(conc.size(-1)),
              "The last dimension of 'conc' must match the number of species");

  auto ww = torch::tensor(
      species_weights,
      torch::TensorOptions().dtype(conc.dtype()).device(conc.device()));

  int ndim = conc.dim();
  std::vector<int64_t> shape(ndim, 1);
  shape.back() = static_cast<int64_t>(species_weights.size());

  return (conc * ww.view(shape)).sum(-1) / conc.sum(-1);
}

}  // namespace harp
