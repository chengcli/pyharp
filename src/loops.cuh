#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

namespace harp {
namespace native {

template <typename func_t>
__global__ void element_kernel(int64_t numel, func_t f) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Shared memory allocation
  extern __shared__ unsigned char memory[];
  char* smem = reinterpret_cast<char*>(memory);

  if (idx < numel) {
    f(idx, smem);
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

template <int Threads, int Arity, typename func_t>
void gpu_mem_kernel(at::TensorIterator& iter, int work_size, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();

  dim3 block(Threads);
  dim3 grid((numel + block.x - 1) / block.x);
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t shared = block.x * work_size;

  // set attribute to allow max dynamic shared memory
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  // query max allowed per-block shared memory
  int max_dynamic_smem = prop.sharedMemPerBlockOptin;
  //printf("max_dynamic_smem = %d\n", max_dynamic_smem);

  auto device_lambda = [=] __device__(int idx, char* smem) {
      auto offsets = offset_calc.get(idx);
      int tid = threadIdx.x;
      f(data.data(), offsets.data(), smem + tid * work_size);
    };

  // request the full size
  auto kernelPtr = element_kernel<decltype(device_lambda)>;
  cudaFuncSetAttribute(
      kernelPtr,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_dynamic_smem);

  if (shared > (size_t)max_dynamic_smem) {
    TORCH_CHECK(false, "Requested shared memory (", shared,
                " bytes) exceeds device maximum (",
                max_dynamic_smem, " bytes).");
  }

  /*std::cout << "block = " << block.x
            << ", grid = " << grid.x
            << ", shared = " << shared
            << ", work_size = " << work_size
            << std::endl;*/

  element_kernel<<<grid, block, shared, stream>>>(numel, device_lambda);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace native
}  // namespace harp
