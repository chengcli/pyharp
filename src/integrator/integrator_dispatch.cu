// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace harp {

void call_average3_cuda(at::TensorIterator& iter, double w1, double w2,
                        double w3) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "average3_cuda", [&]() {
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          return w1 * in1 + w2 * in2 + w3 * in3;
        });
  });
}
}  // namespace harp
