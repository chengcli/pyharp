#pragma once

// torch
#include <torch/torch.h>

// harp
#include <radiation/radiation_band.hpp>

namespace harp {
void write_bin_ascii_header(RadiationBand const& band, std::string fname);

void write_bin_ascii_data(torch::Tensor rad, RadiationBand const& band,
                          std::string fname);
}  // namespace harp
