// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/math/interpolation.hpp>

#include "device_testing.hpp"

namespace {

// Equal including the sign of zero and NaN, which torch::equal treats as
// unequal to itself.
bool same_bits(torch::Tensor const& a, torch::Tensor const& b) {
  if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type()) {
    return false;
  }
  auto bits =
      a.scalar_type() == torch::kFloat64 ? torch::kInt64 : torch::kInt32;
  return torch::equal(a.contiguous().view(bits), b.contiguous().view(bits));
}

// The recursion over tensors is kept for tables that carry gradients, so a
// tracked copy of the table routes the same query through it.
torch::Tensor reference_interpn(std::vector<torch::Tensor> const& query,
                                std::vector<torch::Tensor> const& coords,
                                torch::Tensor const& lookup,
                                bool extrapolate = false) {
  auto tracked = lookup.detach().clone().set_requires_grad(true);
  return harp::interpn(query, coords, tracked, extrapolate).detach();
}

}  // namespace

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

  auto lookup =
      f(x.view({4, 1, 1}), y.view({1, 3, 1}), z.view({1, 1, 2})).unsqueeze(-1);

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
  auto tmp_2d =
      torch::linspace(-35.0, 35.0, ncol * nlyr, opt).view({ncol, nlyr});

  auto narrow = harp::interpn(
      {wave_1d.view({nwave, 1, 1}), lnp_2d.unsqueeze(0), tmp_2d.unsqueeze(0)},
      {kwave, klnp, ktemp}, lookup);

  auto wide =
      harp::interpn({wave_1d.view({nwave, 1, 1}).expand({nwave, ncol, nlyr}),
                     lnp_2d.unsqueeze(0).expand({nwave, ncol, nlyr}),
                     tmp_2d.unsqueeze(0).expand({nwave, ncol, nlyr})},
                    {kwave, klnp, ktemp}, lookup);

  ASSERT_EQ(narrow.sizes(), torch::IntArrayRef({nwave, ncol, nlyr, 2}));
  ASSERT_EQ(wide.sizes(), narrow.sizes());
  EXPECT_TRUE(torch::equal(narrow, wide));
}

TEST(TestInterpolation, testOnNodeQueryMatchesGeneralPath) {
  // A query given at its own coordinate array takes the fast path that skips
  // the zero-weight branch. Handing the same values in a shape that does not
  // match the axis takes the general path. They must agree bit for bit.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  const int nwave = 6, ncol = 3, nlyr = 4;

  auto kwave = torch::linspace(100.0, 600.0, nwave, opt);
  auto klnp = torch::linspace(0.0, 6.0, 5, opt);
  auto lookup = torch::randn({nwave, 5, 2}, opt);
  auto lnp_2d = torch::linspace(0.4, 5.6, ncol * nlyr, opt).view({ncol, nlyr});

  auto fast = harp::interpn({kwave.view({nwave, 1, 1}), lnp_2d.unsqueeze(0)},
                            {kwave, klnp}, lookup);
  auto general =
      harp::interpn({kwave.view({nwave, 1, 1}).expand({nwave, ncol, nlyr}),
                     lnp_2d.unsqueeze(0).expand({nwave, ncol, nlyr})},
                    {kwave, klnp}, lookup);

  ASSERT_EQ(fast.sizes(), torch::IntArrayRef({nwave, ncol, nlyr, 2}));
  EXPECT_TRUE(torch::equal(fast, general));

  // Shifting off the nodes must leave the fast path and change the answer.
  auto shifted =
      harp::interpn({(kwave + 20.0).view({nwave, 1, 1}), lnp_2d.unsqueeze(0)},
                    {kwave, klnp}, lookup);
  EXPECT_FALSE(torch::equal(fast, shifted));
}

TEST(TestInterpolation, testOnNodeVectorRetainsQueryGradients) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto x = torch::tensor({0.0, 1.0, 2.0}, opt);
  auto query = x.clone().set_requires_grad(true);
  auto lookup = x.unsqueeze(-1);

  harp::interpn({query}, {x}, lookup).sum().backward();

  ASSERT_TRUE(query.grad().defined());
  EXPECT_TRUE(torch::equal(query.grad(), torch::tensor({1.0, 1.0, 0.0}, opt)));
}

TEST(TestInterpolation, testOnNodeVectorRetainsCoordinateGradients) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto coord = torch::tensor({0.0, 1.0, 2.0}, opt).set_requires_grad(true);
  auto query = coord.detach().clone();
  auto lookup = query.unsqueeze(-1);

  harp::interpn({query}, {coord}, lookup).sum().backward();

  ASSERT_TRUE(coord.grad().defined());
  EXPECT_TRUE(
      torch::equal(coord.grad(), torch::tensor({-1.0, -1.0, 0.0}, opt)));
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

TEST_P(DeviceTest, FusedMatchesTensorPathOnOpacityQueries) {
  // The shapes an opacity table is queried with: wavenumber over the
  // spectral axis only, pressure and temperature over (column, layer). The
  // table holds -inf where a cross section is zero, as the log tables do.
  torch::manual_seed(0);
  auto opt = torch::TensorOptions().dtype(dtype).device(device);
  const int nwave = 11, ncol = 3, nlyr = 5, nval = 2;

  auto kwave = torch::linspace(100.0, 800.0, nwave, opt);
  auto klnp = torch::linspace(0.0, 6.0, 7, opt);
  auto ktemp = torch::linspace(-40.0, 40.0, 4, opt);
  auto lookup = torch::randn({nwave, 7, 4, nval}, opt);
  lookup[2][3][1][0].fill_(-std::numeric_limits<double>::infinity());
  lookup[5][0][0][1].fill_(-std::numeric_limits<double>::infinity());

  // Pressure runs past both table ends; one temperature lands on a node.
  auto lnp = (torch::rand({ncol, nlyr}, opt) * 8.0 - 1.0).unsqueeze(0);
  auto tempa = torch::rand({ncol, nlyr}, opt) * 100.0 - 50.0;
  tempa[1][2] = ktemp[1];
  tempa = tempa.unsqueeze(0);

  std::vector<torch::Tensor> coords = {kwave, klnp, ktemp};
  for (bool extrapolate : {false, true}) {
    // Wavenumber on its own nodes takes the branch that skips the zero
    // weight; shifted off them, every corner is read.
    for (auto wave :
         {kwave.view({nwave, 1, 1}), (kwave + 7.0).view({nwave, 1, 1})}) {
      std::vector<torch::Tensor> query = {wave, lnp, tempa};
      auto fused = harp::interpn(query, coords, lookup, extrapolate);
      auto reference = reference_interpn(query, coords, lookup, extrapolate);
      ASSERT_EQ(fused.sizes(), torch::IntArrayRef({nwave, ncol, nlyr, nval}));
      EXPECT_FALSE(fused.requires_grad());
      EXPECT_TRUE(same_bits(fused, reference))
          << "extrapolate = " << extrapolate;
    }
  }
}

TEST_P(DeviceTest, FusedMatchesTensorPathInOneAndTwoDimensions) {
  torch::manual_seed(1);
  auto opt = torch::TensorOptions().dtype(dtype).device(device);

  // One dimension, decreasing axis, queries past both ends.
  auto x = torch::tensor({5.0, 4.0, 3.0, 2.0, 1.0}, opt);
  auto table1 = torch::randn({5, 3}, opt);
  auto q1 = torch::linspace(0.0, 6.0, 13, opt);
  EXPECT_TRUE(same_bits(harp::interpn({q1}, {x}, table1),
                        reference_interpn({q1}, {x}, table1)));

  // Two dimensions, fully expanded queries.
  auto y = torch::tensor({10.0, 20.0, 35.0}, opt);
  auto table2 = torch::randn({5, 3, 1}, opt);
  auto qx = torch::rand({4, 6}, opt) * 6.0;
  auto qy = torch::rand({4, 6}, opt) * 30.0 + 5.0;
  EXPECT_TRUE(same_bits(harp::interpn({qx, qy}, {x, y}, table2),
                        reference_interpn({qx, qy}, {x, y}, table2)));
}

TEST_P(DeviceTest, FusedLeavesTrackedQueriesToTensorPath) {
  // A query that is being differentiated must still reach the autograd
  // graph, whatever device it lives on.
  auto opt = torch::TensorOptions().dtype(dtype).device(device);
  auto x = torch::tensor({0.0, 1.0, 2.0}, opt);
  auto lookup = torch::tensor({{1.0}, {3.0}, {2.0}}, opt);
  auto query = torch::tensor({0.25, 1.5}, opt).set_requires_grad(true);

  auto result = harp::interpn({query}, {x}, lookup);
  ASSERT_TRUE(result.requires_grad());
  result.sum().backward();
  ASSERT_TRUE(query.grad().defined());
  EXPECT_TRUE(torch::allclose(query.grad().to(torch::kCPU, torch::kFloat64),
                              torch::tensor({2.0, -1.0}, torch::kFloat64)));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
