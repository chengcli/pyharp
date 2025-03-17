#pragma once

// torch
#include <torch/torch.h>

namespace harp {

torch::Tensor henyey_greenstein(int nmom, torch::Tensor const& g);

}  // namespace harp
