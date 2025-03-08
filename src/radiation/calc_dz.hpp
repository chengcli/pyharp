#pragma once

// torch
#include <torch/torch.h>

namespace harp {

torch::Tensor calc_dz(torch::Tensor pres, torch::Tensor temp,
                      torch::Tensor g_ov_R);

}  // namespace harp
