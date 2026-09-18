// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/math/interpolation.hpp>

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

TEST(TestInterpolation, testGridPointsAreExact) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto x = torch::tensor({1.0, 2.0, 4.0, 8.0}, opt);
  auto y = torch::tensor({-1.0, 0.5, 3.0}, opt);
  auto lookup = torch::arange(4 * 3 * 2, opt).view({4, 3, 2});

  // Query every grid point. Landing on a node must return the tabulated value.
  auto qx = x.unsqueeze(-1).expand({4, 3});
  auto qy = y.unsqueeze(0).expand({4, 3});

  auto result = harp::interpn({qx, qy}, {x, y}, lookup);
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({4, 3, 2}));
  EXPECT_TRUE(torch::allclose(result, lookup));
}

TEST(TestInterpolation, testTrilinearIsReproducedExactly) {
  // Linear interpolation is exact for a function that is itself multilinear,
  // so the result can be checked against the closed form rather than against
  // a previous run.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto x = torch::tensor({0.0, 1.0, 3.0, 7.0}, opt);
  auto y = torch::tensor({10.0, 20.0, 50.0}, opt);
  auto z = torch::tensor({-4.0, -1.0}, opt);

  auto f = [](torch::Tensor const& a, torch::Tensor const& b,
              torch::Tensor const& c) {
    return 2.0 + 3.0 * a - 0.5 * b + 1.5 * c + 0.25 * a * b - 0.75 * a * c +
           0.1 * b * c + 0.05 * a * b * c;
  };

  auto lookup = f(x.view({4, 1, 1}), y.view({1, 3, 1}), z.view({1, 1, 2}))
                    .unsqueeze(-1);

  // Interior query points, deliberately not aligned with any grid node, and
  // shaped so that each dimension is broadcast differently -- the way the
  // opacity tables are queried.
  auto qx = torch::tensor({0.4, 2.2, 6.1}, opt).view({3, 1}).expand({3, 2});
  auto qy = torch::tensor({12.0, 41.0}, opt).view({1, 2}).expand({3, 2});
  auto qz = torch::full({3, 2}, -2.5, opt);

  auto result = harp::interpn({qx, qy, qz}, {x, y, z}, lookup);
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({3, 2, 1}));
  EXPECT_TRUE(torch::allclose(result.squeeze(-1), f(qx, qy, qz), 1e-12, 1e-12));
}

TEST(TestInterpolation, testBroadcastQueryMatchesExpandedQuery) {
  // Opacity tables are queried with a wavenumber that varies only over the
  // spectral axis and a pressure and temperature that vary only over
  // (column, layer). Passing those at their own shape must give exactly the
  // same answer as expanding all three first.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  const int nwave = 7, ncol = 4, nlyr = 5;

  auto kwave = torch::linspace(100.0, 800.0, 9, opt);
  auto klnp = torch::linspace(0.0, 6.0, 6, opt);
  auto ktemp = torch::linspace(-40.0, 40.0, 4, opt);
  auto lookup = torch::randn({9, 6, 4, 2}, opt);

  auto wave_1d = torch::linspace(150.0, 770.0, nwave, opt);
  auto lnp_2d = torch::linspace(0.3, 5.7, ncol * nlyr, opt).view({ncol, nlyr});
  auto tmp_2d = torch::linspace(-35.0, 35.0, ncol * nlyr, opt).view({ncol, nlyr});

  auto narrow = harp::interpn(
      {wave_1d.view({nwave, 1, 1}), lnp_2d.unsqueeze(0), tmp_2d.unsqueeze(0)},
      {kwave, klnp, ktemp}, lookup);

  auto wide = harp::interpn(
      {wave_1d.view({nwave, 1, 1}).expand({nwave, ncol, nlyr}),
       lnp_2d.unsqueeze(0).expand({nwave, ncol, nlyr}),
       tmp_2d.unsqueeze(0).expand({nwave, ncol, nlyr})},
      {kwave, klnp, ktemp}, lookup);

  ASSERT_EQ(narrow.sizes(), torch::IntArrayRef({nwave, ncol, nlyr, 2}));
  ASSERT_EQ(wide.sizes(), narrow.sizes());
  EXPECT_TRUE(torch::equal(narrow, wide));
}

TEST(TestInterpolation, testClampsOutsideTheTable) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto x = torch::tensor({0.0, 1.0, 2.0}, opt);
  auto lookup = torch::tensor({{5.0}, {6.0}, {7.0}}, opt);

  auto query = torch::tensor({-3.0, 0.5, 9.0}, opt);
  auto result = harp::interpn({query}, {x}, lookup);

  // Default is extrapolate=false: the weights are clamped, so queries beyond
  // either end return the edge value.
  EXPECT_NEAR(result[0][0].item<double>(), 5.0, 1e-12);
  EXPECT_NEAR(result[1][0].item<double>(), 5.5, 1e-12);
  EXPECT_NEAR(result[2][0].item<double>(), 7.0, 1e-12);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
