// C/C++
#include <iostream>
#include <vector>

// harp
#include "interpolation.hpp"

namespace harp {

// Recursive helper function for interpolation
torch::Tensor interpn_recur(
    std::vector<torch::Tensor> const& query_coords,
    std::vector<torch::Tensor> const& coords, torch::Tensor const& lookup,
    std::vector<at::indexing::TensorIndex> const& indices, bool extrapolate) {
  int dim = indices.size();
  if (dim == coords.size()) {
    // Base case: Return the interpolated values (final tensor slice)
    return lookup.index(indices);
  }

  // Get current coordinate array
  torch::Tensor coord = coords[dim];
  torch::Tensor query_d = query_coords[dim].flatten();

  // Determine if coordinates are increasing or decreasing
  bool is_increasing = coord[1].item<float>() > coord[0].item<float>();

  // Get searchsorted index
  torch::Tensor search_idx;

  if (is_increasing) {
    search_idx = torch::searchsorted(coord, query_d,
                                     /*out_int32=*/false, /*right=*/true);
  } else {
    search_idx = coord.size(0) - torch::searchsorted(coord.flip(0), query_d,
                                                     /*out_int32=*/false,
                                                     /*right=*/false);
  }

  // Clamp indices within bounds
  auto index_low = torch::clamp(search_idx - 1, 0, coord.size(-1) - 1);
  auto index_high = torch::clamp(index_low + 1, 0, coord.size(-1) - 1);

  // Compute interpolation weights
  auto x0 = coord.index({index_low});
  auto x1 = coord.index({index_high});
  auto diff = x1 - x0;
  diff = torch::where(diff == 0, torch::ones_like(diff),
                      diff);  // Avoid division by zero

  auto weight_high = (query_d - x0) / diff;

  if (!extrapolate) {
    weight_high = torch::clamp(weight_high, 0.0, 1.0);
  }

  auto weight_low = 1.0 - weight_high;

  // Recursively interpolate in the next dimension
  auto indices_low = indices;
  indices_low.push_back(index_low);

  auto interp_low =
      interpn_recur(query_coords, coords, lookup, indices_low, extrapolate);

  auto indices_high = indices;
  indices_high.push_back(index_high);

  auto interp_high =
      interpn_recur(query_coords, coords, lookup, indices_high, extrapolate);

  auto a = interp_low * weight_low.unsqueeze(-1);
  auto b = interp_high * weight_high.unsqueeze(-1);

  // Compute weighted sum
  return interp_low * weight_low.unsqueeze(-1) +
         interp_high * weight_high.unsqueeze(-1);
}

// Wrapper function for interpolation
torch::Tensor interpn(std::vector<torch::Tensor> const& query_coords,
                      std::vector<torch::Tensor> const& coords,
                      torch::Tensor const& lookup, bool extrapolate) {
  // Ensure query coordinates match interpolation dimensions
  TORCH_CHECK(query_coords.size() == coords.size(),
              "Query coordinates must match interpolation dimensions");

  auto nval = lookup.size(-1);
  auto vec = query_coords[0].sizes().vec();
  vec.push_back(nval);

  // Perform recursive interpolation
  return interpn_recur(query_coords, coords, lookup, {}, extrapolate).view(vec);
}

}  // namespace harp
