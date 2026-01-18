#pragma once

// C/C++
#include <cstdlib>

namespace harp {

// Solves a tridiagonal system using the Thomas algorithm (TDMA)
template <typename T>
void dtridgl(int n, const T *a, const T *b, const T *c, const T *d, T *x,
             char *mem, int &offset) {
  T *cp = (T *)get_mem(n, sizeof(T), mem, &offset);
  T *dp = (T *)get_mem(n, sizeof(T), mem, &offset);

  if (!cp || !dp) {
    // Handle memory allocation failure
    exit(EXIT_FAILURE);
  }

  // First row
  cp[0] = c[0] / b[0];
  dp[0] = d[0] / b[0];

  // Forward sweep
  for (int i = 1; i < n; ++i) {
    T denom = b[i] - a[i] * cp[i - 1];
    if (denom == 0.0) denom = 1e-12;  // Avoid division by zero
    cp[i] = (i < n - 1) ? c[i] / denom : 0.0;
    dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
  }

  // Back substitution
  x[n - 1] = dp[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    x[i] = dp[i] - cp[i] * x[i + 1];
  }
}

}  // namespace harp
