// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// harp
#include "interpn.h"

namespace harp {

template <int N>
void call_interpn_cpu(at::TensorIterator& iter, torch::Tensor kdata,
                      torch::Tensor axis, torch::Tensor dims, int nval) {
  TORCH_CHECK(N >= nval, "N must be greater than or equal to nval");

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "interpn_cpu", [&] {
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto coord = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        interpn<N>(out, coord, kdata.data_ptr<scalar_t>(),
                   axis.data_ptr<scalar_t>(), dims.data_ptr<int64_t>(),
                   dims.size(0), nval);
      }
    });
  });
}

template void call_interpn_cpu<1>(at::TensorIterator& iter, torch::Tensor kdata,
                                  torch::Tensor axis, torch::Tensor dims,
                                  int nval);

template void call_interpn_cpu<2>(at::TensorIterator& iter, torch::Tensor kdata,
                                  torch::Tensor axis, torch::Tensor dims,
                                  int nval);

template void call_interpn_cpu<3>(at::TensorIterator& iter, torch::Tensor kdata,
                                  torch::Tensor axis, torch::Tensor dims,
                                  int nval);

}  // namespace harp
