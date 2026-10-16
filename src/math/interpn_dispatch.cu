// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <c10/cuda/CUDAGuard.h>

// harp
#include <harp/loops.cuh>

#include "interpn_dispatch.hpp"
#include "interpn_impl.h"

namespace harp {

template <int NDIM>
void interpn_fused_cuda_impl(at::TensorIterator& iter,
                             at::Tensor const& lookup, unsigned on_nodes) {
  constexpr int Arity = 2 + 4 * NDIM;
  InterpnTable geom{};
  for (int d = 0; d < NDIM; ++d) geom.stride[d] = lookup.stride(d);

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "interpn_fused_cuda", [&] {
    auto table = lookup.data_ptr<scalar_t>();
    native::gpu_kernel<Arity>(
        iter, [=] GPU_LAMBDA(char* const data[Arity],
                             unsigned int strides[Arity]) {
          char* ptr[Arity];
          for (int k = 0; k < Arity; ++k) ptr[k] = data[k] + strides[k];
          interpn_element<scalar_t, NDIM>(ptr, table, geom, on_nodes);
        });
  });
}

void call_interpn_fused_cuda(at::TensorIterator& iter,
                             at::Tensor const& lookup, int ndim,
                             unsigned on_nodes) {
  at::cuda::CUDAGuard device_guard(iter.device());

  switch (ndim) {
    case 1:
      interpn_fused_cuda_impl<1>(iter, lookup, on_nodes);
      break;
    case 2:
      interpn_fused_cuda_impl<2>(iter, lookup, on_nodes);
      break;
    case 3:
      interpn_fused_cuda_impl<3>(iter, lookup, on_nodes);
      break;
    case 4:
      interpn_fused_cuda_impl<4>(iter, lookup, on_nodes);
      break;
    default:
      TORCH_CHECK(false, "interpn: fused kernel takes 1 to 4 dimensions, got ",
                  ndim);
  }
}

}  // namespace harp

namespace at::native {

REGISTER_CUDA_DISPATCH(call_interpn_fused, &harp::call_interpn_fused_cuda);

}  // namespace at::native
