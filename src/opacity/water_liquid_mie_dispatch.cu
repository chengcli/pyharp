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
                                double molecular_weight, int max_order) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_water_liquid_mie_cuda", [&] {
    using ComplexScalar = Complex<scalar_t>;
    size_t const work_size = 3 * static_cast<size_t>(max_order) *
                             sizeof(ComplexScalar);
    native::gpu_chunk_kernel<8, 9>(
        iter, work_size,
        [=] GPU_LAMBDA(char* const data[9], unsigned int strides[9],
                       char* work) {
          auto extinction = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto single_scattering_albedo =
              reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto g = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto conc = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto wave = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto re = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
          auto density = reinterpret_cast<scalar_t*>(data[6] + strides[6]);
          auto real = reinterpret_cast<scalar_t*>(data[7] + strides[7]);
          auto imag = reinterpret_cast<scalar_t*>(data[8] + strides[8]);
          auto* work_ptr = reinterpret_cast<ComplexScalar*>(work);
          auto const properties = water_liquid_mie_properties(
              *conc, *wave, *re, *density, *real, *imag,
              static_cast<scalar_t>(molecular_weight), work_ptr, max_order);
          *extinction = properties.extinction;
          *single_scattering_albedo = properties.single_scattering_albedo;
          *g = properties.g;
        });
  });
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_water_liquid_mie,
                       &harp::call_water_liquid_mie_cuda);

}  // namespace at::native
