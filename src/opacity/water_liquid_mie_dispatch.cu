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

void call_water_liquid_mie_efficiency_cuda(at::TensorIterator& iter,
                                           int max_order) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(
      iter.dtype(), "call_water_liquid_mie_efficiency_cuda", [&] {
        using ComplexScalar = Complex<scalar_t>;
        size_t const work_size = 3 * static_cast<size_t>(max_order) *
                                 sizeof(ComplexScalar);
        native::gpu_chunk_kernel<8>(
            iter, work_size,
            [=] GPU_LAMBDA(char* const data[8], unsigned int strides[8],
                           char* work) {
              auto qext = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto qsca = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              auto g = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
              auto status = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
              auto wave = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
              auto re = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
              auto real = reinterpret_cast<scalar_t*>(data[6] + strides[6]);
              auto imag = reinterpret_cast<scalar_t*>(data[7] + strides[7]);
              auto* work_ptr = reinterpret_cast<ComplexScalar*>(work);
              auto const mie = mie_efficiency_device(
                  *real, *imag, mie_size_parameter(*re, *wave), work_ptr,
                  max_order);
              *qext = mie.qext;
              *qsca = mie.qsca;
              *g = mie.g;
              *status = static_cast<scalar_t>(mie.status);
            });
      });
}

void call_water_liquid_mie_assemble_cuda(at::TensorIterator& iter,
                                         double molecular_weight) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(
      iter.dtype(), "call_water_liquid_mie_assemble_cuda", [&] {
        native::gpu_kernel<10>(
            iter, [=] GPU_LAMBDA(char* const data[10], unsigned int strides[10]) {
              auto extinction =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto single_scattering_albedo =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              auto g = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
              auto conc = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
              auto re = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
              auto density = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
              auto qext = reinterpret_cast<scalar_t*>(data[6] + strides[6]);
              auto qsca = reinterpret_cast<scalar_t*>(data[7] + strides[7]);
              auto gq = reinterpret_cast<scalar_t*>(data[8] + strides[8]);
              auto status = reinterpret_cast<scalar_t*>(data[9] + strides[9]);
              MieEfficiencyDevice<scalar_t> const mie{
                  *qext, *qsca, *gq, static_cast<int>(*status)};
              auto const properties = water_liquid_mie_properties_from(
                  *conc, *re, *density,
                  static_cast<scalar_t>(molecular_weight), mie);
              *extinction = properties.extinction;
              *single_scattering_albedo = properties.single_scattering_albedo;
              *g = properties.g;
            });
      });
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_water_liquid_mie_efficiency,
                       &harp::call_water_liquid_mie_efficiency_cuda);
REGISTER_CUDA_DISPATCH(call_water_liquid_mie_assemble,
                       &harp::call_water_liquid_mie_assemble_cuda);

}  // namespace at::native
