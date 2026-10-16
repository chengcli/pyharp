// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAGuard.h>

// harp
#include <harp/loops.cuh>

#include "cloud_kernels_dispatch.hpp"
#include "cloud_kernels_impl.h"

namespace harp {

void call_fu_ice_fill_cuda(at::TensorIterator& iter, at::Tensor const& table,
                           int nmom, bool delta_scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "fu_ice_fill_cuda", [&] {
    auto rows = table.data_ptr<scalar_t>();
    native::gpu_kernel<5>(
        iter, [=] GPU_LAMBDA(char* const data[5], unsigned int strides[5]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto invalid = reinterpret_cast<uint8_t*>(data[1] + strides[1]);
          auto wave = *reinterpret_cast<int64_t const*>(data[2] + strides[2]);
          auto d = *reinterpret_cast<scalar_t const*>(data[3] + strides[3]);
          auto iwc = *reinterpret_cast<scalar_t const*>(data[4] + strides[4]);
          *invalid = fu_ice_row(out, rows + wave * fu::kWidth, d, iwc, nmom,
                                delta_scale);
        });
  });
}

void call_cloud_switch_fill_cuda(at::TensorIterator& iter, int nmom) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "cloud_switch_fill_cuda", [&] {
    native::gpu_kernel<3>(
        iter, [=] GPU_LAMBDA(char* const data[3], unsigned int strides[3]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto ice = reinterpret_cast<scalar_t const*>(data[1] + strides[1]);
          auto liquid = reinterpret_cast<scalar_t const*>(data[2] + strides[2]);
          cloud_switch_row(out, ice, liquid, nmom);
        });
  });
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_fu_ice_fill, &harp::call_fu_ice_fill_cuda);
REGISTER_CUDA_DISPATCH(call_cloud_switch_fill,
                       &harp::call_cloud_switch_fill_cuda);

}  // namespace at::native
