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
  std::cout << "dim = " << dim << std::endl;
  if (dim == coords.size()) {
    // Base case: Return the interpolated values (final tensor slice)
    std::cout << "lookup = " << lookup << std::endl;
    return lookup.index(indices);
  }

  // Get current coordinate array
  torch::Tensor coord = coords[dim];
  torch::Tensor query_d = query_coords[dim].flatten();

  std::cout << "query_d = " << query_d << std::endl;

  // Get searchsorted index
  auto search_idx = torch::searchsorted(coord, query_d,
                                        /*out_int32=*/false, /*right=*/false);

  std::cout << "search_idx = " << search_idx << std::endl;

  // Clamp indices within bounds
  auto index_low = torch::clamp(search_idx - 1, 0, coord.size(-1) - 2);
  auto index_high = index_low + 1;

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

  std::cout << "weight_low = " << weight_low << std::endl;

  // Recursively interpolate in the next dimension
  std::cout << "index_low = " << index_low << std::endl;
  std::cout << "index_high = " << index_high << std::endl;

  // get interp dims
  // auto vec = query_coords[0].sizes().vec();
  // auto vec = lookup.sizes().vec();
  // vec[dim] = query_coords[0].numel();
  auto indices_low = indices;
  indices_low.push_back(index_low);

  auto interp_low =
      interpn_recur(query_coords, coords, lookup, indices_low, extrapolate);
  // interp_low = interp_low.view(vec);

  auto indices_high = indices;
  indices_high.push_back(index_high);

  auto interp_high =
      interpn_recur(query_coords, coords, lookup, indices_high, extrapolate);
  // interp_high = interp_high.view(vec);

  std::cout << "1) interp_low = " << interp_low << std::endl;
  std::cout << "2) weight_low = " << weight_low << std::endl;
  std::cout << "3) interp_high = " << interp_high << std::endl;
  std::cout << "4) weight_high = " << weight_high << std::endl;

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
