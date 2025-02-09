#pragma once

#include <torch/torch.h>

namespace harp {

//! read the "weights" variable from a netCDF file
torch::Tensor read_weights_rfm(std::string const& filename);

}  // namespace harp
