#pragma once

// C/C++
#include <cstdlib>

// base
#include <configure.h>

namespace harp {

// Solves a tridiagonal system using the Thomas algorithm (TDMA)
template <typename T>
DISPATCH_MACRO void dtridgl(int n, const T *a, const T *b, T *c, T *d, T *x) {
  // First row
  c[0] = c[0] / b[0];
  d[0] = d[0] / b[0];

  // Forward sweep
  for (int i = 1; i < n; ++i) {
    T denom = b[i] - a[i] * c[i - 1];
    if (denom == 0.0) denom = 1e-12;  // Avoid division by zero
    c[i] = (i < n - 1) ? c[i] / denom : 0.0;
    d[i] = (d[i] - a[i] * d[i - 1]) / denom;
  }

  // Back substitution
  x[n - 1] = d[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    x[i] = d[i] - c[i] * x[i + 1];
  }
}

}  // namespace harp
