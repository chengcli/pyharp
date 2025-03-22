#pragma once

// torch
#include <torch/torch.h>

namespace harp {

// ---------------------------------------------------------------------
// 1) Factorization: A = L*U  (Thomas algorithm, "compressed" storage)
// ---------------------------------------------------------------------
void tridiag_lu(torch::Tensor& a, torch::Tensor& b, const torch::Tensor& c);

// ---------------------------------------------------------------------
// 2) Solve A x = f for each batch, using the LU factorization
//    (a_, b_) returned by tridiag_lu.
// ---------------------------------------------------------------------
void tridiag_solve(torch::Tensor& f,          // right-hand side, shape (..., n)
                   const torch::Tensor& a_,   // subdiagonal (L)
                   const torch::Tensor& b_,   // main diagonal (U)
                   const torch::Tensor& c_);  // superdiagonal (U)

// ---------------------------------------------------------------------
// 3) Helper to apply a tridiagonal matrix (a,b,c) to x => f
//    for testing. A is of shape (m,n,n) conceptually, but we store in
//    compressed form: a, b, c each (m,n).
// ---------------------------------------------------------------------
torch::Tensor tridiag_matmul2d_slow(const torch::Tensor& a,
                                    const torch::Tensor& b,
                                    const torch::Tensor& c,
                                    const torch::Tensor& x);

}  // namespace harp
