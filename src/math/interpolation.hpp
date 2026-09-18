#pragma once

// torch
#include <torch/torch.h>

namespace harp {

//! Multidimensional linear interpolation
/*!
 * The query tensors only have to broadcast against one another, they do not
 * have to be the same shape. A dimension whose query is constant along some
 * axis can be passed with that axis of size one, and the search for that
 * dimension then runs on the small tensor rather than on the broadcast result.
 *
 * \param query_coords Query coordinates, len = ndim, mutually broadcastable
 * \param coords Coordinate arrays, len = ndim, each tensor has shape (nx1,),
 * (nx2,) ...
 * \param lookup Lookup tensor (nx1, nx2, ..., nval)
 * \return Interpolated values, shape broadcast(query_coords) + (nval,)
 */
torch::Tensor interpn(std::vector<torch::Tensor> const& query_coords,
                      std::vector<torch::Tensor> const& coords,
                      torch::Tensor const& lookup, bool extrapolate = false);

template <int N>
void call_interpn_cpu(at::TensorIterator& iter, at::Tensor kdata,
                      at::Tensor axis, at::Tensor dims, int nval);

template <int N>
void call_interpn_cuda(at::TensorIterator& iter, at::Tensor kdata,
                       at::Tensor axis, at::Tensor dims, int nval);

}  // namespace harp
