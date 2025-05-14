#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! Read a 2D data table into a Torch tensor
torch::Tensor read_data_tensor(std::string const& fname);

}  // namespace harp
