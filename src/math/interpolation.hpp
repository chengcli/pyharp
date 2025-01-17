#pragma once

// torch
#include <ATen/TensorIterator.h>

namespace harp {

template <int N>
void call_interpn_cpu(at::TensorIterator& iter, at::Tensor kdata,
                      at::Tensor axis, at::Tensor dims, int ndim);

template <int N>
void call_interpn_cuda(at::TensorIterator& iter, at::Tensor kdata,
                       at::Tensor axis, at::Tensor dims, int ndim);

}  // namespace harp
