#pragma once

// C/C++
#include <cstdint>
#include <type_traits>

// base
#include <configure.h>

namespace harp {

//! Table strides a fused interpolation kernel carries by value.
struct InterpnTable {
  int64_t stride[4];
};

//! low * wl + high * wh, rounded like torch's separate mul and add kernels,
//! so the fused result matches the tensor expression bit for bit.
template <typename T>
DISPATCH_MACRO T interpn_blend(T low, T wl, T high, T wh) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, double>::value) {
    return __dadd_rn(__dmul_rn(low, wl), __dmul_rn(high, wh));
  } else {
    return __fadd_rn(__fmul_rn(low, wl), __fmul_rn(high, wh));
  }
#else
  T const a = low * wl;
  T const b = high * wh;
  return a + b;
#endif
}

//! Blend the table corners of one query, dimension D outermost, in the
//! same order as the recursion over tensors in interpolation.cpp.
/*!
 * \p low and \p high hold each dimension's bracketing offsets into the
 * table, already multiplied by that dimension's stride. Bit d of
 * \p on_nodes says that dimension d's query sits on its own nodes, so only
 * the low branch is read; multiplying the high branch by its zero weight
 * would turn a -inf table entry into NaN.
 */
template <typename T, int NDIM, int D = 0>
DISPATCH_MACRO T interpn_corners(T const* table, int64_t offset,
                                 int64_t const* low, int64_t const* high,
                                 T const* wl, T const* wh, unsigned on_nodes) {
  if constexpr (D == NDIM) {
    return table[offset];
  } else {
    T const vlow = interpn_corners<T, NDIM, D + 1>(table, offset + low[D], low,
                                                   high, wl, wh, on_nodes);
    if (on_nodes & (1u << D)) return vlow;
    T const vhigh = interpn_corners<T, NDIM, D + 1>(
        table, offset + high[D], low, high, wl, wh, on_nodes);
    return interpn_blend(vlow, wl[D], vhigh, wh[D]);
  }
}

//! One output element. \p ptr holds this element's address in every
//! iterator operand: the output, the value index along the trailing axis,
//! then per table dimension its low index, high index, low weight and high
//! weight.
template <typename T, int NDIM>
DISPATCH_MACRO void interpn_element(char* const* ptr, T const* table,
                                    InterpnTable const& geom,
                                    unsigned on_nodes) {
  auto out = reinterpret_cast<T*>(ptr[0]);
  int64_t const value = *reinterpret_cast<int64_t const*>(ptr[1]);

  int64_t low[NDIM];
  int64_t high[NDIM];
  T wl[NDIM];
  T wh[NDIM];
  for (int d = 0; d < NDIM; ++d) {
    char* const* p = ptr + 2 + 4 * d;
    low[d] = *reinterpret_cast<int64_t const*>(p[0]) * geom.stride[d];
    high[d] = *reinterpret_cast<int64_t const*>(p[1]) * geom.stride[d];
    wl[d] = *reinterpret_cast<T const*>(p[2]);
    wh[d] = *reinterpret_cast<T const*>(p[3]);
  }

  *out = interpn_corners<T, NDIM>(table, value, low, high, wl, wh, on_nodes);
}

}  // namespace harp
