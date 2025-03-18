// external
#include <gtest/gtest.h>

// harp
#include <harp/math/trapezoid.hpp>

TEST(Trapezoid, Basic) {
  // Test the trapezoidal rule with a simple function
  torch::Tensor x = torch::tensor({0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
  torch::Tensor y = torch::tensor({0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0});

  // Expected result: integral of x^2 from 0 to 3
  double expected = (3.0 * 3.0 * 3.0) / 3.0;

  // Compute the integral using the trapezoidal rule
  auto result = harp::trapezoid(y, x);

  std::cout << "result: " << result << std::endl;

  // Check that the result is close to the expected value
  EXPECT_NEAR(result.item<double>(), expected, 0.2);
}

TEST(Trapezoid, multidimension) {
  // Test the trapezoidal rule with a multidimensional tensor
  torch::Tensor x = torch::tensor({0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
  torch::Tensor y = torch::tensor({{0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0},
                                   {0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0},
                                   {0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0}});

  std::cout << "y: " << y << std::endl;

  // Compute the integral using the trapezoidal rule
  auto result = harp::trapezoid(y, x);

  std::cout << "result: " << result << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
