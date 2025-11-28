// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// harp
#include "integrator_dispatch.hpp"

namespace harp {

void call_average3_cpu(at::TensorIterator& iter, double w1, double w2,
                       double w3) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_averag3_cpu", [&] {
    at::native::cpu_kernel(
        iter,
        [&](scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          return w1 * in1 + w2 * in2 + w3 * in3;
        },
        grain_size);
  });
}

void call_average3_mps(at::TensorIterator& iter, double w1, double w2,
                       double w3) {
  auto out = iter.output();
  auto u1 = iter.input(0);
  auto u2 = iter.input(1);
  auto u3 = iter.input(2);
  torch::add_out(out, w1 * u1 + w2 * u2, w3 * u3);
}

}  // namespace harp

namespace at::native {

DEFINE_DISPATCH(call_average3);
REGISTER_ALL_CPU_DISPATCH(call_average3, &harp::call_average3_cpu);
REGISTER_MPS_DISPATCH(call_average3, &harp::call_average3_mps);

}  // namespace at::native
