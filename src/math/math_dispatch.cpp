// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// harp
#include <math/interpn.h>

namespace harp {

template <int N>
void call_interpn_cpu(at::TensorIterator& iter, torch::Tensor dims, int ndim) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "interpn_cpu", [&] {
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto coord = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto kdata = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto axis = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
        interpn<N>(out, coord, kdata, axis, dims.data_ptr<int64_t>(), ndim);
      }
    });
  });
}

template void call_interpn_cpu<2>(at::TensorIterator& iter, torch::Tensor dims,
                                  int ndim);
template void call_interpn_cpu<3>(at::TensorIterator& iter, torch::Tensor dims,
                                  int ndim);

}  // namespace harp
