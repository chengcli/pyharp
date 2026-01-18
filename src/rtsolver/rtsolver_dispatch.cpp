// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// harp
#include "rtsolver_dispatch.hpp"
#include "toon_mckay89_longwave_impl.h"
#include "toon_mckay89_shortwave_impl.h"

namespace harp {

void call_toon89_sw_cpu(at::TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_toon89_sw_cpu", [&] {
    int nlay = at::native::ensure_nonempty_size(iter.input(1), -2);
    int grain_size = iter.numel() / at::get_num_threads();
    int mem_size = toon89_sw_space<scalar_t>(nlay);
    char *work = new char[mem_size];

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto prop = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto umu0 = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            auto fbeam = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
            auto albedo =
                reinterpret_cast<scalar_t *>(data[4] + i * strides[4]);
            toon_mckay89_shortwave(nlay, *fbeam, umu0, prop, *albedo, out,
                                   work);
          }
        },
        grain_size);

    delete[] work;
  });
}

void call_toon89_lw_cpu(at::TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_toon89_lw_cpu", [&] {
    int nlay = at::native::ensure_nonempty_size(iter.input(1), -2);
    int grain_size = iter.numel() / at::get_num_threads();
    int mem_size = toon89_sw_space<scalar_t>(nlay);
    char *work = new char[mem_size];

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto prop = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto albedo =
                reinterpret_cast<scalar_t *>(data[4] + i * strides[4]);
            auto be = reinterpret_cast<scalar_t *>(data[5] + i * strides[5]);
            toon_mckay89_longwave(nlay, be, prop, *albedo, out, work);
          }
        },
        grain_size);

    delete[] work;
  });
}

}  // namespace harp

namespace at::native {

DEFINE_DISPATCH(call_toon89_lw);
DEFINE_DISPATCH(call_toon89_sw);

REGISTER_ALL_CPU_DISPATCH(call_toon89_lw, &harp::call_toon89_lw_cpu);
REGISTER_ALL_CPU_DISPATCH(call_toon89_sw, &harp::call_toon89_sw_cpu);

}  // namespace at::native
