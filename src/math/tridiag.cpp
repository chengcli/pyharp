// harp
#include "tridiag.hpp"

namespace harp {

void tridiag_lu(torch::Tensor& a, torch::Tensor& b, const torch::Tensor& c) {
  // Check that a, b, c have the same shape
  TORCH_CHECK(a.sizes() == b.sizes(), "Shape mismatch between a and b");
  TORCH_CHECK(b.sizes() == c.sizes(), "Shape mismatch between b and c");

  // Dimensions
  auto n = a.size(-1);  // dimension of each tridiagonal system

  // Thomas forward pass in a batched manner:
  //   b_[:, 0] = b[:, 0]                (initial pivot)
  //   For k in 1..(n-1):
  //       a_[:, k] = a_[:, k] / b_[:, k-1]        (subdiagonal L)
  //       b_[:, k] = b_[:, k] - a_[:, k] * c[:, k-1]  (update pivot)
  for (int64_t k = 1; k < n; k++) {
    a.select(-1, k) /= b.select(-1, k - 1);
    b.select(-1, k) -= a.select(-1, k) * c.select(-1, k - 1);
  }

  // Now:
  //   a holds subdiagonal of L (L's main diagonal is implicitly 1)
  //   b holds the main diagonal of U
}

void tridiag_solve(torch::Tensor& f,         // right-hand side, shape (m, n)
                   const torch::Tensor& a_,  // subdiagonal (L)
                   const torch::Tensor& b_,  // main diagonal (U)
                   const torch::Tensor& c_   // superdiagonal (U)
) {
  // Check shapes
  TORCH_CHECK(a_.sizes() == b_.sizes(), "Shape mismatch between a_ and b_");
  TORCH_CHECK(b_.sizes() == c_.sizes(), "Shape mismatch between b_ and c_");
  TORCH_CHECK(f.sizes() == a_.sizes(), "Batch size mismatch between a_ and f");

  auto n = a_.size(-1);  // dimension

  // -----------------------------------------------------------------
  // Forward substitution: Solve L y = f
  //   L is unit lower-triangular: diag(L)=1, subdiag(L)=a_
  //   => y[:, 0] = f[:, 0]
  //      y[:, k] = f[:, k] - a_[:, k]*y[:, k-1],  k=1..(n-1)
  // -----------------------------------------------------------------
  for (int64_t k = 1; k < n; k++) {
    f.select(-1, k) -= a_.select(-1, k) * f.select(-1, k - 1);
  }

  // -----------------------------------------------------------------
  // Back substitution: Solve U x = y
  //   U is upper-triangular: diag(U)=b_, superdiag(U)=c_
  //   => x[:, n-1] = y[:, n-1]/b_[:, n-1]
  //      x[:, k]   = (y[:, k] - c_[:, k]*x[:, k+1]) / b_[:, k],  k=(n-2)..0
  // -----------------------------------------------------------------
  f.select(-1, n - 1) /= b_.select(-1, n - 1);
  for (int64_t k = n - 2; k >= 0; k--) {
    f.select(-1, k) =
        (f.select(-1, k) - c_.select(-1, k) * f.select(-1, k + 1)) /
        b_.select(-1, k);
  }
}

torch::Tensor tridiag_matmul2d_slow(const torch::Tensor& a,
                                    const torch::Tensor& b,
                                    const torch::Tensor& c,
                                    const torch::Tensor& x) {
  // x shape is (m, n), output f also (m, n).
  auto options = x.options();
  auto m = x.size(0);
  auto n = x.size(-1);

  auto f = torch::zeros_like(x);

  // f[:, j] = b[:, j]*x[:, j]
  //          + c[:, j]*x[:, j+1] (if j+1 < n)
  //          + a[:, j]*x[:, j-1] (if j-1 >= 0)
  // We'll just loop:
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < n; j++) {
      float val = b[i][j].item<float>() * x[i][j].item<float>();

      if (j + 1 < n) {
        val += c[i][j].item<float>() * x[i][j + 1].item<float>();
      }
      if (j - 1 >= 0) {
        val += a[i][j].item<float>() * x[i][j - 1].item<float>();
      }

      f[i][j] = val;
    }
  }
  return f;
}

}  // namespace harp
