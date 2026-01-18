// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAGuard.h>

// harp
#include <harp/loops.cuh>
#include "rtsolver_dispatch.hpp"
#include "toon_mckay89_longwave_impl.h"
#include "toon_mckay89_shortwave_impl.h"

namespace harp {

void call_toon89_sw_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_toon89_sw_cuda", [&] {
    int nlay = at::native::ensure_nonempty_size(iter.input(1), -2);
    int mem_size = toon89_sw_space<scalar_t>(nlay);

    native::gpu_mem_kernel<128, 5>(
        iter, [=] GPU_LAMBDA(
          char* const data[5], unsigned int strides[5], char *work) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto prop = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto umu0 = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto fbeam = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto albedo = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          toon_mckay89_shortwave(nlay, *fbeam, umu0, prop, *albedo, out, work);
        });
  });
}

void call_toon89_lw_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_toon89_lw_cuda", [&] {
    int nlay = at::native::ensure_nonempty_size(iter.input(1), -2);
    int mem_size = toon89_sw_space<scalar_t>(nlay);

    native::gpu_mem_kernel<128, 4>(
        iter, [=] GPU_LAMBDA(
          char* const data[4], unsigned int strides[4], char *work) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto prop = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto be = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto albedo = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          toon_mckay89_longwave(nlay, be, prop, *albedo, out, work);
        });
  });
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_toon89_lw, &harp::call_toon89_lw_cuda);
REGISTER_CUDA_DISPATCH(call_toon89_sw, &harp::call_toon89_sw_cuda);

} // namespace at::native
