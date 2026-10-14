#pragma once
#include <torch/torch.h>

// Shared by both CIA reference gates; preserve the candidate-scaled bound.
inline torch::Tensor cia_gradient_failures(torch::Tensor const& reference,
                                         torch::Tensor const& candidate) {
  auto bound = 1.0e-14 + 3.0e-5 * candidate.abs();
  return (~torch::isfinite(reference)) | (~torch::isfinite(candidate)) |
         ((reference - candidate).abs() > bound);
}
