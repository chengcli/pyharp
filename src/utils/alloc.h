#pragma once

// C/C++
#include <cstddef>
#include <cstdint>
#include <type_traits>

// base
#include <configure.h>

namespace harp {

DISPATCH_MACRO inline uintptr_t align_up(uintptr_t p, size_t a) {
  // a must be power of two; works for 4, 8, 16, ...
  return (p + (a - 1)) & ~(a - 1);
}

template <typename U>
DISPATCH_MACRO inline U* alloc_from(char*& cursor, size_t count) {
  uintptr_t p = reinterpret_cast<uintptr_t>(cursor);
  p = align_up(p, alignof(U));
  U* out = reinterpret_cast<U*>(p);
  cursor = reinterpret_cast<char*>(p + count * sizeof(U));
  return out;
}

template <typename T>
size_t toon89_sw_space(int nlay) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };

  int nlev = nlay + 1;
  bump(alignof(T), nlev * sizeof(T));        // dir
  bump(alignof(T), nlev * sizeof(T));        // tau
  bump(alignof(T), nlev * sizeof(T));        // cum_trans
  bump(alignof(T), nlev * sizeof(T));        // tau_in
  bump(alignof(T), nlay * sizeof(T));        // dtau
  bump(alignof(T), nlay * sizeof(T));        // mu_zm
  bump(alignof(T), nlay * sizeof(T));        // w0
  bump(alignof(T), nlay * sizeof(T));        // hg
  bump(alignof(T), nlay * sizeof(T));        // gam
  bump(alignof(T), nlay * sizeof(T));        // Am
  bump(alignof(T), nlay * sizeof(T));        // Ap
  bump(alignof(T), nlay * sizeof(T));        // Cpm1
  bump(alignof(T), nlay * sizeof(T));        // Cmm1
  bump(alignof(T), nlay * sizeof(T));        // Cp
  bump(alignof(T), nlay * sizeof(T));        // Cm
  bump(alignof(T), nlay * sizeof(T));        // Ep
  bump(alignof(T), nlay * sizeof(T));        // Em
  bump(alignof(T), nlay * sizeof(T));        // E1
  bump(alignof(T), nlay * sizeof(T));        // E2
  bump(alignof(T), nlay * sizeof(T));        // E3
  bump(alignof(T), nlay * sizeof(T));        // E4
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Af
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Bf
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Cf
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Df
  bump(alignof(T), (2 * nlay) * sizeof(T));  // xk
  bump(alignof(T), nlay * sizeof(T));        // xk1
  bump(alignof(T), nlay * sizeof(T));        // xk2

  return bytes;
}

// Element i of a thread's array sits i * stride elements after element 0.
// When the threads of a chunk get consecutive base addresses and a stride
// equal to the chunk size, a warp touching the same element reads one
// contiguous run instead of 32 scattered words. With stride 1 this is the
// plain contiguous layout.
template <typename T>
struct StridedArray {
  T* data;
  int stride;
  DISPATCH_MACRO T& operator[](int i) const {
    return data[static_cast<ptrdiff_t>(i) * stride];
  }
};

template <typename U>
DISPATCH_MACRO inline StridedArray<U> alloc_strided(char*& cursor,
                                                    size_t count, int stride) {
  U* out = reinterpret_cast<U*>(cursor);
  cursor += count * static_cast<size_t>(stride) * sizeof(U);
  return StridedArray<U>{out, stride};
}

template <typename T>
size_t toon89_lw_space(int nlay) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };
  int nlev = nlay + 1;

  bump(alignof(T), nlay * sizeof(T));        // dtau
  bump(alignof(T), nlev * sizeof(T));        // tau
  bump(alignof(T), nlay * sizeof(T));        // w0
  bump(alignof(T), nlay * sizeof(T));        // hg
  bump(alignof(T), nlay * sizeof(T));        // B0
  bump(alignof(T), nlay * sizeof(T));        // B1
  bump(alignof(T), nlay * sizeof(T));        // lam
  bump(alignof(T), nlay * sizeof(T));        // gam
  bump(alignof(T), nlay * sizeof(T));        // alp
  bump(alignof(T), nlay * sizeof(T));        // term
  bump(alignof(T), nlay * sizeof(T));        // Cpm1
  bump(alignof(T), nlay * sizeof(T));        // Cmm1
  bump(alignof(T), nlay * sizeof(T));        // Cp
  bump(alignof(T), nlay * sizeof(T));        // Cm
  bump(alignof(T), nlay * sizeof(T));        // Ep
  bump(alignof(T), nlay * sizeof(T));        // Em
  bump(alignof(T), nlay * sizeof(T));        // E1
  bump(alignof(T), nlay * sizeof(T));        // E2
  bump(alignof(T), nlay * sizeof(T));        // E3
  bump(alignof(T), nlay * sizeof(T));        // E4
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Af
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Bf
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Cf
  bump(alignof(T), (2 * nlay) * sizeof(T));  // Df
  bump(alignof(T), (2 * nlay) * sizeof(T));  // xkk
  bump(alignof(T), nlay * sizeof(T));        // xk1
  bump(alignof(T), nlay * sizeof(T));        // xk2
  bump(alignof(T), nlay * sizeof(T));        // g
  bump(alignof(T), nlay * sizeof(T));        // h
  bump(alignof(T), nlay * sizeof(T));        // xj
  bump(alignof(T), nlay * sizeof(T));        // xk
  bump(alignof(T), nlay * sizeof(T));        // alpha1
  bump(alignof(T), nlay * sizeof(T));        // alpha2
  bump(alignof(T), nlay * sizeof(T));        // sigma1
  bump(alignof(T), nlay * sizeof(T));        // sigma2
  bump(alignof(T), nlay * sizeof(T));        // em1
  bump(alignof(T), nlev * sizeof(T));        // lw_up_g
  bump(alignof(T), nlev * sizeof(T));        // lw_down_g

  return bytes;
}
}  // namespace harp
