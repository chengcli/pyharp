#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

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
void gpu_chunk_kernel(at::TensorIterator& iter, int work_size, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  // devide numel into Chunk parts to reduce memory usage
  // allocate working memory pool
  char* d_workspace = nullptr;

  // workspace size per chunk
  int chunks = Chunks > numel ? numel : Chunks;
  int base = numel / chunks;
  int rem  = numel % chunks;

  size_t workspace_bytes = work_size * (base + (rem > 0 ? 1 : 0));
  cudaMalloc(&d_workspace, workspace_bytes);

  int chunk_start = 0;

  for (int n = 0; n < chunks; n++) {
    int64_t chunk_numel = base + (n < rem ? 1 : 0);
    int64_t chunk_end = chunk_start + chunk_numel;  // exclusive

    dim3 block(64);
    dim3 grid((chunk_numel + block.x - 1) / block.x);

    auto device_lambda = [=] __device__(int idx, char* work) {
        auto offsets = offset_calc.get(idx + chunk_start);
        f(data.data(), offsets.data(), work + idx * work_size);
      };

    /*std::cout << "chunk = " << n
              << ", chunk_start = " << chunk_start
              << ", chunk_end = " << chunk_end
              << ", chunk_numel = " << chunk_numel
              << ", block = " << block.x
              << ", grid = " << grid.x
              << ", work_size = " << work_size
              << std::endl;*/

    element_kernel<<<grid, block>>>(chunk_numel, device_lambda, d_workspace);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaDeviceSynchronize();

    chunk_start = chunk_end;
  }

  // free working memory pool
  cudaFree(d_workspace);
}

}  // namespace native
}  // namespace harp
