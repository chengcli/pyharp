// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/math/tridiag.hpp>

// ---------------------------------------------------------------------
// Example "main" to test factorization + solve
// ---------------------------------------------------------------------
TEST(TestTridiagSolver, test1) {
  // Create a small batch of 2 tridiagonal systems, each of dimension 4
  // a, b, c have shape (2, 4).
  auto options = torch::TensorOptions().dtype(torch::kFloat);

  // Subdiagonal
  torch::Tensor a = torch::tensor(
      {{0.0f, 1.0f, 2.0f, 3.0f}, {0.0f, 4.0f, 5.0f, 6.0f}}, options);

  // Main diagonal
  torch::Tensor b = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f}, {7.0f, 8.0f, 9.0f, 10.0f}}, options);

  // Superdiagonal
  torch::Tensor c = torch::tensor(
      {{0.5f, 1.5f, 2.5f, 0.0f}, {3.5f, 4.5f, 5.5f, 0.0f}}, options);

  // Pick a "true" solution x_true
  torch::Tensor x_true = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}}, options);

  // Compute f = A x_true to have a known right-hand side
  torch::Tensor f = harp::tridiag_matmul2d_slow(a, b, c, x_true);

  // Factor A => L, U
  harp::tridiag_lu(a, b, c);

  // Solve for x: A x = f
  auto x_solved = f.clone();
  harp::tridiag_solve(x_solved, a, b, c);

  std::cout << "\n== Tridiagonal Systems Test ==\n\n";
  std::cout << "a (subdiag):\n" << a << "\n";
  std::cout << "b (diag):\n" << b << "\n";
  std::cout << "c (superdiag):\n" << c << "\n\n";

  std::cout << "x_true:\n" << x_true << "\n";
  std::cout << "f = A*x_true:\n" << f << "\n";

  std::cout << "\nLU Factorization results:\n";
  std::cout << "a_factor (subdiag of L):\n" << a << "\n";
  std::cout << "b_factor (diag of U):\n" << b << "\n";
  std::cout << "c_factor (superdiag of U):\n" << c << "\n";

  std::cout << "\nSolved x:\n" << x_solved << "\n";

  // Compare x_solved with x_true
  auto diff = (x_solved - x_true).abs().max().item<float>();
  EXPECT_NEAR(diff, 0.0f, 1e-6f);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
