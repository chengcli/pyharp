// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <math/interpolation.hpp>

TEST(TestInterpolation, test1DIncreasing) {
  // Coordinate arrays
  std::vector<torch::Tensor> coords = {
      torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0}),  // X-coordinates
  };

  // Lookup data dimension (5,1)
  torch::Tensor lookup = torch::tensor({{1.0}, {4.0}, {9.0}, {16.0}, {25.0}});

  // Query coordinates (10,) representing the x-coordinates to interpolate
  std::vector<torch::Tensor> query_coords = {torch::linspace(0.0, 6.0, 10)};

  std::cout << "query = " << query_coords[0] << std::endl;

  // Perform interpolation
  torch::Tensor result = harp::interpn(query_coords, coords, lookup);
  std::cout << "Interpolated Values:\n" << result << std::endl;
}

TEST(TestInterpolation, test1DDecreasing) {
  // Coordinate arrays
  std::vector<torch::Tensor> coords = {
      torch::tensor({5.0, 4.0, 3.0, 2.0, 1.0}),  // X-coordinates
  };

  // Lookup data dimension (5,1)
  torch::Tensor lookup = torch::tensor({{25.0}, {16.0}, {9.0}, {4.0}, {1.0}});

  // Query coordinates (10,) representing the x-coordinates to interpolate
  std::vector<torch::Tensor> query_coords = {torch::linspace(0.0, 6.0, 10)};

  std::cout << "query = " << query_coords[0] << std::endl;

  // Perform interpolation
  torch::Tensor result = harp::interpn(query_coords, coords, lookup);
  std::cout << "Interpolated Values:\n" << result << std::endl;
}

TEST(TestInterpolation, testND) {
  // Coordinate arrays
  std::vector<torch::Tensor> coords = {
      torch::tensor({1.0, 2.0, 3.0}),    // X-coordinates
      torch::tensor({10.0, 20.0, 30.0})  // Y-coordinates
  };

  // Lookup data (3,3,2) where the last dimension represents 2 variables at each
  // (x,y)
  torch::Tensor lookup =
      torch::tensor({{{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}},
                     {{4.0, 40.0}, {5.0, 50.0}, {6.0, 60.0}},
                     {{7.0, 70.0}, {8.0, 80.0}, {9.0, 90.0}}});

  // Query coordinates (2, 2, 2) representing (x,y) coordinates to interpolate
  std::vector<torch::Tensor> query_coords = {
      torch::tensor({{2.5, 3.5}, {0., -1.}}),    // X-coordinates
      torch::tensor({{15.0, 25.0}, {0., -10.}})  // Y-coordinates
  };

  // Perform interpolation
  torch::Tensor result = harp::interpn(query_coords, coords, lookup);
  std::cout << "Interpolated Values:\n" << result << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
