#pragma once

// torch
#include <torch/torch.h>

namespace harp {

torch::Tensor henyey_greenstein(int nmom, torch::Tensor const& g);

torch::Tensor double_henyey_greenstein(int nmom, torch::Tensor const& ff,
                                       torch::Tensor const& g1,
                                       torch::Tensor const& g2);

}  // namespace harp
