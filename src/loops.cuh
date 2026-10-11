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

// One chunk's workspace is kept within this many bytes. The chunks run one
// after another on the stream and reuse the same workspace, so this bounds
// the memory a kernel takes however large the problem is.
constexpr size_t kChunkWorkspaceBytes = static_cast<size_t>(2) << 30;

// Number of chunks that keeps one chunk's workspace within
// kChunkWorkspaceBytes. Fewer chunks means more threads per launch.
inline int64_t workspace_chunks(int64_t numel, size_t work_size) {
  if (work_size == 0) return 1;
  int64_t per_chunk = static_cast<int64_t>(kChunkWorkspaceBytes / work_size);
  if (per_chunk < 1) per_chunk = 1;
  return (numel + per_chunk - 1) / per_chunk;
}

// Runs f once per element, in as many chunks as the workspace budget needs.
// Contiguous: each thread's workspace is one block, f(data, offsets, work).
// Interleaved: the threads of a chunk share their workspace so that element
// i of thread t sits at byte (i * chunk_numel + t) * elem_size, and f also
// gets the stride chunk_numel, f(data, offsets, work, stride). A warp
// touching the same element then reads one contiguous run.
// Calls the functor with or without the workspace stride. A separate type
// rather than if constexpr, because an extended __device__ lambda may not
// first-capture a variable inside a constexpr-if branch.
template <bool Interleaved>
struct ChunkCall;

template <>
struct ChunkCall<false> {
  template <typename func_t, typename data_t, typename offset_t>
  __device__ static void run(func_t const& f, data_t data, offset_t offsets,
                             char* work, int /*stride*/) {
    f(data, offsets, work);
  }
};

template <>
struct ChunkCall<true> {
  template <typename func_t, typename data_t, typename offset_t>
  __device__ static void run(func_t const& f, data_t data, offset_t offsets,
                             char* work, int stride) {
    f(data, offsets, work, stride);
  }
};

template <int Arity, bool Interleaved, typename func_t>
void gpu_chunk_kernel_impl(at::TensorIterator& iter, size_t work_size,
                           size_t elem_size, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);
  if (Interleaved) {
    TORCH_CHECK(elem_size > 0 && work_size % elem_size == 0,
                "interleaved workspace must hold whole elements");
  }

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.numel();
  if (numel == 0) return;

  int64_t chunks = workspace_chunks(numel, work_size);
  int64_t base = numel / chunks;
  int64_t rem = numel % chunks;

  size_t const max_chunk_numel =
      static_cast<size_t>(base + (rem > 0 ? 1 : 0));
  TORCH_CHECK(max_chunk_numel <=
                  static_cast<size_t>(std::numeric_limits<int>::max()),
              "GPU chunk too large");
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

  // Bytes between the workspaces of neighbouring threads.
  size_t const step = Interleaved ? elem_size : work_size;

  int64_t chunk_start = 0;
  for (int64_t n = 0; n < chunks; n++) {
    int64_t chunk_numel = base + (n < rem ? 1 : 0);
    int const stride = static_cast<int>(chunk_numel);

    dim3 block(64);
    dim3 grid((chunk_numel + block.x - 1) / block.x);

    auto device_lambda = [=] __device__(int idx, char* work) {
      auto offsets = offset_calc.get(idx + chunk_start);
      ChunkCall<Interleaved>::run(f, data.data(), offsets.data(),
                                  work + idx * step, stride);
    };

    // Stream ordering lets every chunk safely reuse the same workspace without
    // a device-wide synchronization between launches.
    element_kernel<<<grid, block, 0, stream>>>(chunk_numel, device_lambda,
                                               d_workspace);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    chunk_start += chunk_numel;
  }
}

template <int Arity, typename func_t>
void gpu_chunk_kernel(at::TensorIterator& iter, size_t work_size,
                      const func_t& f) {
  gpu_chunk_kernel_impl<Arity, false>(iter, work_size, 1, f);
}

template <int Arity, typename func_t>
void gpu_chunk_kernel_interleaved(at::TensorIterator& iter, size_t work_size,
                                  size_t elem_size, const func_t& f) {
  gpu_chunk_kernel_impl<Arity, true>(iter, work_size, elem_size, f);
}

}  // namespace native
}  // namespace harp
