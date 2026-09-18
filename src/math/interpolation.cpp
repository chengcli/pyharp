// C/C++
#include <iostream>
#include <vector>

// harp
#include "interpolation.hpp"

namespace harp {

namespace {

//! Bracketing indices and linear weights for one interpolation dimension.
struct AxisWeights {
  torch::Tensor index_low;
  torch::Tensor index_high;
  torch::Tensor weight_low;
  torch::Tensor weight_high;
};

AxisWeights locate_on_axis(torch::Tensor const& coord,
                           torch::Tensor const& query_d, bool extrapolate) {
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

  AxisWeights out;

  // Clamp indices within bounds
  out.index_low = torch::clamp(search_idx - 1, 0, coord.size(-1) - 1);
  out.index_high = torch::clamp(out.index_low + 1, 0, coord.size(-1) - 1);

  // Compute interpolation weights
  auto x0 = coord.index({out.index_low});
  auto x1 = coord.index({out.index_high});
  auto diff = x1 - x0;
  diff = torch::where(diff == 0, torch::ones_like(diff),
                      diff);  // Avoid division by zero

  out.weight_high = (query_d - x0) / diff;

  if (!extrapolate) {
    out.weight_high = torch::clamp(out.weight_high, 0.0, 1.0);
  }

  out.weight_low = 1.0 - out.weight_high;

  // The recursion below broadcasts the weights against the trailing value
  // dimension of the lookup table, so give them that shape once here rather
  // than on every visit.
  out.weight_low = out.weight_low.unsqueeze(-1);
  out.weight_high = out.weight_high.unsqueeze(-1);

  return out;
}

// Recursive helper function for interpolation
torch::Tensor interpn_recur(
    std::vector<AxisWeights> const& axes, torch::Tensor const& lookup,
    std::vector<at::indexing::TensorIndex> const& indices) {
  int dim = indices.size();
  if (dim == axes.size()) {
    // Base case: Return the interpolated values (final tensor slice)
    return lookup.index(indices);
  }

  auto const& axis = axes[dim];

  // Recursively interpolate in the next dimension
  auto indices_low = indices;
  indices_low.push_back(axis.index_low);

  auto interp_low = interpn_recur(axes, lookup, indices_low);

  auto indices_high = indices;
  indices_high.push_back(axis.index_high);

  auto interp_high = interpn_recur(axes, lookup, indices_high);

  // Compute weighted sum
  return interp_low * axis.weight_low + interp_high * axis.weight_high;
}

}  // namespace

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

  // Bracket every dimension once. The weights depend only on that dimension's
  // query, not on the path taken through the earlier dimensions, so computing
  // them inside the recursion repeated the search and the weight arithmetic
  // 2^dim times for dimension dim.
  std::vector<AxisWeights> axes;
  axes.reserve(coords.size());
  for (size_t dim = 0; dim < coords.size(); ++dim) {
    axes.push_back(
        locate_on_axis(coords[dim], query_coords[dim].flatten(), extrapolate));
  }

  // Perform recursive interpolation
  return interpn_recur(axes, lookup, {}).view(vec);
}

}  // namespace harp
