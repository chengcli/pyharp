#pragma once

// torch
#include <torch/torch.h>

namespace harp {

torch::Tensor read_data_tensor(std::string const& fname);

}  // namespace harp
