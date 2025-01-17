// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// harp
#include <math/interpn.h>

namespace harp {

void call_interpn_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "interpn_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ncloud = nhydro - nvapor - index::ICY;

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto wli = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto wri = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto gammad = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
        auto cv_ratio_m1 =
            reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
        auto inv_mu_ratio_m1 =
            reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
        interpn(out, coor, data, axis, len, ndim);
      }
    });
  });
}

}  // namespace harp
