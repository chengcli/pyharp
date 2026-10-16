// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAGuard.h>

// harp
#include <harp/loops.cuh>

#include "band_accumulate_dispatch.hpp"
#include "band_accumulate_impl.h"

namespace harp {

void call_accumulate_opacity_cuda(at::TensorIterator& iter, int nprop) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "accumulate_opacity_cuda", [&] {
    native::gpu_kernel<2>(
        iter, [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2]) {
          auto prop = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto k = reinterpret_cast<scalar_t const*>(data[1] + strides[1]);
          accumulate_opacity_row(prop, k, nprop);
        });
  });
}

void call_finish_band_cuda(at::TensorIterator& iter, int nprop,
                           double albedo_cap) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "finish_band_cuda", [&] {
    auto const cap = static_cast<scalar_t>(albedo_cap);
    native::gpu_kernel<2>(
        iter, [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2]) {
          auto prop = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto dz = *reinterpret_cast<scalar_t const*>(data[1] + strides[1]);
          finish_band_row(prop, nprop, dz, cap);
        });
  });
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_accumulate_opacity,
                       &harp::call_accumulate_opacity_cuda);
REGISTER_CUDA_DISPATCH(call_finish_band, &harp::call_finish_band_cuda);

}  // namespace at::native
