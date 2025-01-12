// torch
#include <torch/torch.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/ReduceOpsUtils.h>

// harp
#include "disort_impl.h"

namespace harp {

void call_disort_cpu(at::TensorIterator& iter, int rank_in_column,
                     std::vector<disort_state> &ds,
                     std::vector<disort_output> &ds_out) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "disort_cpu", [&] {
    auto nprop = at::native::ensure_nonempty_size(iter.output(), -1);

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out  = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto prop = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto ftoa = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto temf = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
        auto iwave = reinterpret_cast<int64_t*>(data[4] + i * strides[4]);
        disort_impl(out, prop, ftoa, temf, rank_in_column, ds[*iwave], ds_out[*iwave], nprop);
      }
    });
  });
}

} // namespace harp
