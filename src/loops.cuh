#pragma once

// torch
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// C/C++
#include <limits>

namespace harp {
namespace native {

template <typename func_t>
__global__ void element_kernel(int64_t numel, func_t f, char *work) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  if (idx < numel) {
    f(idx, work);
  }
}

template <int Arity, typename func_t>
void gpu_kernel(at::TensorIterator& iter, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  at::native::launch_legacy_kernel<128, 1>(numel,
      [=] __device__(int idx) {
      auto offsets = offset_calc.get(idx);
      f(data.data(), offsets.data());
    });
}

template <int Chunks, int Arity, typename func_t>
void gpu_chunk_kernel(at::TensorIterator& iter, size_t work_size,
                      const func_t& f) {
  static_assert(Chunks > 0, "gpu_chunk_kernel requires at least one chunk");
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();
  if (numel == 0) return;

  // Divide numel into chunks so that one reusable workspace bounds memory use.
  int64_t chunks = Chunks > numel ? numel : Chunks;
  int64_t base = numel / chunks;
  int64_t rem = numel % chunks;

  size_t const max_chunk_numel =
      static_cast<size_t>(base + (rem > 0 ? 1 : 0));
  TORCH_CHECK(work_size == 0 ||
                  max_chunk_numel <=
                      std::numeric_limits<size_t>::max() / work_size,
              "GPU workspace size overflow");
  size_t const workspace_bytes = work_size * max_chunk_numel;
  TORCH_CHECK(
      workspace_bytes <=
          static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "GPU workspace is too large");
  auto workspace = at::empty(
      {static_cast<int64_t>(workspace_bytes)},
      iter.output(0).options().dtype(at::kByte));
  char* d_workspace = static_cast<char*>(workspace.data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream(iter.device().index());

  int64_t chunk_start = 0;

  for (int64_t n = 0; n < chunks; n++) {
    int64_t chunk_numel = base + (n < rem ? 1 : 0);
    int64_t chunk_end = chunk_start + chunk_numel;  // exclusive

    dim3 block(64);
    dim3 grid((chunk_numel + block.x - 1) / block.x);

    auto device_lambda = [=] __device__(int idx, char* work) {
      auto offsets = offset_calc.get(idx + chunk_start);
      f(data.data(), offsets.data(), work + idx * work_size);
    };

    // Stream ordering lets every chunk safely reuse the same workspace without
    // a device-wide synchronization between launches.
    element_kernel<<<grid, block, 0, stream>>>(chunk_numel, device_lambda,
                                               d_workspace);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    chunk_start = chunk_end;
  }
}

}  // namespace native
}  // namespace harp
