#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! numerator / denominator where denominator > 0, else 0; denominator gains
//! trailing dimensions until it broadcasts against numerator
inline torch::Tensor divide_where_positive(torch::Tensor numerator,
                                           torch::Tensor denominator) {
  while (denominator.dim() < numerator.dim()) {
    denominator = denominator.unsqueeze(-1);
  }
  auto const positive = denominator > 0.0;
  auto const safe_denominator =
      torch::where(positive, denominator, torch::ones_like(denominator));
  return torch::where(positive, numerator / safe_denominator,
                      torch::zeros_like(numerator));
}

}  // namespace harp
