#pragma once

// C/C++
#include <type_traits>

// base
#include <configure.h>

namespace harp {

// One IEEE operation each, rounded to nearest. On the device these are the
// intrinsics that stop nvcc from contracting a multiply and an add into a
// fused multiply-add, so a kernel that mirrors a chain of torch elementwise
// ops reproduces the same bits.

template <typename T>
DISPATCH_MACRO T rn_add(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, double>::value) {
    return __dadd_rn(a, b);
  } else {
    return __fadd_rn(a, b);
  }
#else
  return a + b;
#endif
}

template <typename T>
DISPATCH_MACRO T rn_sub(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, double>::value) {
    return __dsub_rn(a, b);
  } else {
    return __fsub_rn(a, b);
  }
#else
  return a - b;
#endif
}

template <typename T>
DISPATCH_MACRO T rn_mul(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, double>::value) {
    return __dmul_rn(a, b);
  } else {
    return __fmul_rn(a, b);
  }
#else
  return a * b;
#endif
}

template <typename T>
DISPATCH_MACRO T rn_div(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, double>::value) {
    return __ddiv_rn(a, b);
  } else {
    return __fdiv_rn(a, b);
  }
#else
  return a / b;
#endif
}

}  // namespace harp
