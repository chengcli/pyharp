// C/C++
#include <iostream>
#include <mutex>
#include <unordered_map>
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
  //! Query sits exactly on the nodes, so the high branch has zero weight.
  bool on_nodes = false;
};

//! Whether a coordinate axis increases along its only dimension, cached by
//! tensor identity. Opacity tables register their coordinate axes as
//! construction-time buffers that never change afterward, but interpn() is a
//! free function with no per-instance state to remember that in, so this
//! keeps a small process-wide table instead. A held reference to the tensor
//! keeps its storage alive so a later, unrelated tensor cannot reuse the same
//! address and be mistaken for it; sizes/dtype/device are re-checked on every
//! lookup as a cheap (no device sync) guard against exactly that.
class AxisDirectionCache {
 public:
  bool is_increasing(torch::Tensor const& coord) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = table_.find(coord.data_ptr());
    if (it != table_.end()) {
      auto const& [cached_tensor, cached_value] = it->second;
      if (cached_tensor.sizes() == coord.sizes() &&
          cached_tensor.scalar_type() == coord.scalar_type() &&
          cached_tensor.device() == coord.device()) {
        return cached_value;
      }
    }

    bool const increasing = coord[1].item<float>() > coord[0].item<float>();
    table_[coord.data_ptr()] = {coord, increasing};
    return increasing;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<void const*, std::pair<torch::Tensor, bool>> table_;
};

AxisDirectionCache& axis_direction_cache() {
  static AxisDirectionCache cache;
  return cache;
}

//! Combines two pointer hashes; std::unordered_map has no built-in hash for
//! std::pair.
struct PointerPairHash {
  size_t operator()(std::pair<void const*, void const*> const& p) const {
    size_t const h1 = std::hash<void const*>{}(p.first);
    size_t const h2 = std::hash<void const*>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

//! Is every query value exactly one of the tabulated coordinates, in order?
/*!
 * Opacity bands are built from the wavenumber axis of the table they read, so
 * that dimension is usually queried at its own nodes. Interpolating there is a
 * gather with weight one, and the other branch of the recursion is multiplied
 * by zero and thrown away, which for a 3D table is half of the corner reads.
 *
 * The shape/dtype check above is metadata only and never reaches the table
 * data, so it is free. It also means a query whose shape does not match the
 * axis (pressure and temperature queried over (ncol, nlyr) against a much
 * shorter table axis, say) returns before the `torch::equal` below, which is
 * the only line here that syncs the device. In practice that line only runs
 * for a band's own wavenumber grid queried against the matching table axis,
 * and both are construction-time buffers that never change, so the result is
 * cached by tensor identity just like the axis direction above.
 */
class OnNodesCache {
 public:
  bool query_lies_on_nodes(torch::Tensor const& coord,
                           torch::Tensor const& query_d) {
    auto q = query_d.squeeze();
    if (q.sizes() != coord.sizes() || q.scalar_type() != coord.scalar_type()) {
      return false;
    }

    auto const key = std::make_pair(coord.data_ptr(), q.data_ptr());
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = table_.find(key);
    if (it != table_.end()) {
      auto const& [cached_coord, cached_q, cached_value] = it->second;
      if (cached_coord.sizes() == coord.sizes() &&
          cached_coord.scalar_type() == coord.scalar_type() &&
          cached_coord.device() == coord.device() &&
          cached_q.sizes() == q.sizes() &&
          cached_q.scalar_type() == q.scalar_type() &&
          cached_q.device() == q.device()) {
        return cached_value;
      }
    }

    bool const on_nodes = torch::equal(q, coord);
    table_[key] = {coord, q, on_nodes};
    return on_nodes;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::pair<void const*, void const*>,
                     std::tuple<torch::Tensor, torch::Tensor, bool>,
                     PointerPairHash>
      table_;
};

OnNodesCache& on_nodes_cache() {
  static OnNodesCache cache;
  return cache;
}

AxisWeights locate_on_axis(torch::Tensor const& coord,
                           torch::Tensor const& query_d, bool extrapolate) {
  // Determine if coordinates are increasing or decreasing
  bool is_increasing = axis_direction_cache().is_increasing(coord);

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
  // than on every visit. Everything keeps the query's own shape, so a query
  // that is constant along some axis costs nothing along that axis.
  out.weight_low = out.weight_low.unsqueeze(-1);
  out.weight_high = out.weight_high.unsqueeze(-1);
  out.on_nodes = on_nodes_cache().query_lies_on_nodes(coord, query_d);

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

  // The high branch carries weight zero, so skip it and the multiply.
  if (axis.on_nodes) {
    return interp_low;
  }

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

  // Bracket every dimension once. The weights depend only on that dimension's
  // query, not on the path taken through the earlier dimensions, so computing
  // them inside the recursion repeated the search and the weight arithmetic
  // 2^dim times for dimension dim.
  //
  // Each dimension is bracketed at the shape it was handed in. The queries
  // only have to broadcast against each other, so a caller that varies
  // pressure over (ncol, nlyr) and wavenumber over (nwave,) can pass
  // (1, ncol, nlyr) and (nwave, 1, 1) instead of expanding both to the full
  // (nwave, ncol, nlyr); the gather below broadcasts them and the search runs
  // on the small tensors. Passing fully expanded queries still works, it just
  // repeats the search along the expanded axes.
  std::vector<AxisWeights> axes;
  axes.reserve(coords.size());
  for (size_t dim = 0; dim < coords.size(); ++dim) {
    axes.push_back(locate_on_axis(coords[dim], query_coords[dim], extrapolate));
  }

  // Perform recursive interpolation
  return interpn_recur(axes, lookup, {});
}

}  // namespace harp
