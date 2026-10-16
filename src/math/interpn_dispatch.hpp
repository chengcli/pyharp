#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

//! \brief Gather the 2^ndim table corners of every query and blend them.
/*!
 * The iterator runs over the output, shape broadcast(queries) + (nval,).
 * Its operands are the output, the value index along the trailing axis,
 * then for each table dimension its bracketing low index, high index, low
 * weight and high weight, each at the shape of that dimension's query.
 * \p lookup is the contiguous table (n1, ..., nndim, nval). Bit d of
 * \p on_nodes says that dimension d's query sits on its own nodes, so its
 * high branch is skipped rather than multiplied by zero.
 */
using interpn_fused_fn = void (*)(at::TensorIterator& iter,
                                  at::Tensor const& lookup, int ndim,
                                  unsigned on_nodes);

DECLARE_DISPATCH(interpn_fused_fn, call_interpn_fused);

}  // namespace at::native
