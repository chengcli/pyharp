// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

namespace harp {
void call_average3_cpu(at::TensorIterator& iter, double w1, double w2,
                       double w3) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "averag3_cpu", [&] {
    at::native::cpu_kernel(
        iter, [&](scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          return w1 * in1 + w2 * in2 + w3 * in3;
        });
  });
}
}  // namespace harp
