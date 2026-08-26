// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAGuard.h>

// harp
#include <harp/loops.cuh>

#include "water_liquid_mie_dispatch.hpp"
#include "water_liquid_mie_impl.h"

namespace harp {

void call_water_liquid_mie_cuda(at::TensorIterator& iter,
                                double molecular_weight, int nmom,
                                int max_order) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_water_liquid_mie_cuda", [&] {
    using ComplexScalar = Complex<scalar_t>;
    size_t const work_size = 3 * static_cast<size_t>(max_order) *
                             sizeof(ComplexScalar);
    native::gpu_chunk_kernel<8, 8>(
        iter, work_size,
        [=] GPU_LAMBDA(char* const data[8], unsigned int strides[8],
                       char* work) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto prop = reinterpret_cast<int64_t*>(data[1] + strides[1]);
          auto conc = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto wave = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto re = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto density = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
          auto real = reinterpret_cast<scalar_t*>(data[6] + strides[6]);
          auto imag = reinterpret_cast<scalar_t*>(data[7] + strides[7]);
          auto* work_ptr = reinterpret_cast<ComplexScalar*>(work);
          auto const value = water_liquid_mie_property(
              *prop, *conc, *wave, *re, *density, *real, *imag,
              static_cast<scalar_t>(molecular_weight), nmom, work_ptr,
              max_order);
          *out = value;
        });
  });
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_water_liquid_mie,
                       &harp::call_water_liquid_mie_cuda);

}  // namespace at::native
