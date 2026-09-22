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

TEST(TestInterpolation, testAxisDirectionCacheDistinguishesDifferentAxes) {
  // locate_on_axis() caches each coordinate axis's monotonic direction by
  // tensor identity to avoid a host sync on every call, since opacity tables
  // register their axes once at construction and never change them. Exercise
  // two differently-ordered axes of the same shape/dtype back to back, and
  // reuse each one across repeated calls, to make sure the cache never
  // conflates distinct axes or serves a stale answer.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto increasing = torch::tensor({1.0, 2.0, 3.0, 4.0}, opt);
  auto decreasing = torch::tensor({4.0, 3.0, 2.0, 1.0}, opt);
  auto lookup = torch::tensor({{1.0}, {2.0}, {3.0}, {4.0}}, opt);
  auto query = torch::tensor({1.5, 2.5, 3.5}, opt);

  auto from_increasing = harp::interpn({query}, {increasing}, lookup);
  auto from_decreasing = harp::interpn({query}, {decreasing}, lookup);
  EXPECT_TRUE(torch::allclose(from_increasing,
                              torch::tensor({{1.5}, {2.5}, {3.5}}, opt)));
  EXPECT_TRUE(torch::allclose(from_decreasing,
                              torch::tensor({{3.5}, {2.5}, {1.5}}, opt)));

  // Repeat both, in reverse order, to exercise the cache-hit path.
  auto from_decreasing_again = harp::interpn({query}, {decreasing}, lookup);
  auto from_increasing_again = harp::interpn({query}, {increasing}, lookup);
  EXPECT_TRUE(torch::equal(from_decreasing, from_decreasing_again));
  EXPECT_TRUE(torch::equal(from_increasing, from_increasing_again));
}

TEST(TestInterpolation, testOnNodesCacheDistinguishesDifferentQueries) {
  // query_lies_on_nodes() caches its result by (axis, query) identity once
  // their shapes and dtypes match closely enough to reach torch::equal, since
  // that pairing is exactly a band's own wavenumber grid queried against the
  // matching table axis -- both construction-time buffers that never change.
  // Interleave the on-nodes query with an off-nodes query of the identical
  // shape, and repeat each, to make sure the cache never conflates the two or
  // serves a stale answer.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  const int nwave = 5, ncol = 2, nlyr = 2;
  auto kwave = torch::linspace(100.0, 500.0, nwave, opt);
  auto klnp = torch::linspace(0.0, 6.0, 4, opt);
  auto lookup = torch::randn({nwave, 4, 2}, opt);
  auto lnp_2d = torch::linspace(0.4, 5.6, ncol * nlyr, opt).view({ncol, nlyr});

  auto on_nodes_query = kwave.view({nwave, 1, 1});
  auto off_nodes_query = (kwave + 10.0).view({nwave, 1, 1});

  auto on_nodes_general =
      harp::interpn({on_nodes_query.expand({nwave, ncol, nlyr}),
                     lnp_2d.unsqueeze(0).expand({nwave, ncol, nlyr})},
                    {kwave, klnp}, lookup);

  auto run_on_nodes = [&] {
    return harp::interpn({on_nodes_query, lnp_2d.unsqueeze(0)}, {kwave, klnp},
                         lookup);
  };
  auto run_off_nodes = [&] {
    return harp::interpn({off_nodes_query, lnp_2d.unsqueeze(0)}, {kwave, klnp},
                         lookup);
  };

  auto on1 = run_on_nodes();
  auto off1 = run_off_nodes();
  auto on2 = run_on_nodes();
  auto off2 = run_off_nodes();

  EXPECT_TRUE(torch::equal(on1, on_nodes_general));
  EXPECT_TRUE(torch::equal(on1, on2));
  EXPECT_TRUE(torch::equal(off1, off2));
  EXPECT_FALSE(torch::equal(on1, off1));
}

TEST(TestInterpolation, testCachesDistinguishStridedViewsOfOneBuffer) {
  // Both identity caches must treat strides as part of a tensor's identity.
  // Two views of one buffer can share the first element, shape, dtype, and
  // device while reading different values: here an increasing and a
  // decreasing axis, and an on-nodes and an off-nodes query.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);

  // Axis direction: {2, 3} at stride 1 versus {2, 1} at stride 2.
  auto axis_buffer = torch::tensor({2.0, 3.0, 1.0}, opt);
  auto increasing = axis_buffer.slice(0, 0, 2, 1);
  auto decreasing = axis_buffer.slice(0, 0, 3, 2);
  ASSERT_EQ(increasing.data_ptr(), decreasing.data_ptr());
  auto lookup = torch::tensor({{10.0}, {20.0}}, opt);
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto up = harp::interpn({torch::tensor({2.25}, opt)}, {increasing}, lookup);
    auto down =
        harp::interpn({torch::tensor({1.25}, opt)}, {decreasing}, lookup);
    EXPECT_NEAR(up[0][0].item<double>(), 12.5, 1e-12);
    EXPECT_NEAR(down[0][0].item<double>(), 17.5, 1e-12);
  }

  // On-nodes: the axis values at stride 2 versus an interleaved off-nodes
  // sequence at stride 1, both starting at the same element.
  const int nwave = 4;
  auto kwave = torch::linspace(100.0, 400.0, nwave, opt);
  auto query_buffer = torch::stack({kwave, kwave + 10.0}, 1).flatten();
  auto on_nodes = query_buffer.slice(0, 0, 2 * nwave, 2);
  auto off_nodes = query_buffer.slice(0, 0, nwave, 1);
  ASSERT_EQ(on_nodes.data_ptr(), off_nodes.data_ptr());
  ASSERT_TRUE(torch::equal(on_nodes, kwave));
  ASSERT_FALSE(torch::equal(off_nodes, kwave));
  auto table = torch::randn({nwave, 2}, opt);
  auto run = [&](torch::Tensor const& q) {
    return harp::interpn({q.unsqueeze(-1)}, {kwave}, table);
  };
  for (int repeat = 0; repeat < 2; ++repeat) {
    // A fresh contiguous copy has its own identity, so it never shares a
    // cache entry with the view and serves as an independent reference.
    EXPECT_TRUE(torch::equal(run(on_nodes), run(on_nodes.clone())));
    EXPECT_TRUE(torch::equal(run(off_nodes), run(off_nodes.clone())));
  }
}

TEST(TestInterpolation, testCachesDoNotRetainTensors) {
  // The caches are process-wide and see per-call temporaries, so they must
  // hold only weak references: a query passed to interpn() must be freed as
  // soon as the caller drops it, or every step of a model would leak one.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto kwave = torch::linspace(100.0, 400.0, 4, opt);
  auto table = torch::randn({4, 2}, opt);
  auto const kwave_owners = kwave.storage().use_count();

  auto released_query = [&] {
    auto query = kwave.clone().unsqueeze(-1);
    harp::interpn({query}, {kwave}, table);
    harp::interpn({query}, {kwave}, table);
    return query.storage().getWeakStorageImpl();
  }();

  EXPECT_TRUE(released_query.expired());
  EXPECT_EQ(kwave.storage().use_count(), kwave_owners);
}

TEST(TestInterpolation, testCachesSurviveAddressReuse) {
  // Since nothing is retained, a freed axis's address may be handed to the
  // next allocation. Make that deterministic: wrap one caller-owned buffer in
  // a fresh tensor (fresh storage, same address) on every iteration, with the
  // axis direction flipped each time, so a stale entry left under the reused
  // address would produce the wrong bracket.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto lookup = torch::tensor({{10.0}, {20.0}, {30.0}}, opt);
  std::vector<double> buffer(3);
  void const* first_address = nullptr;
  for (int repeat = 0; repeat < 8; ++repeat) {
    bool const increasing = repeat % 2 == 0;
    // Overwrite in place so the buffer keeps its address.
    for (int i = 0; i < 3; ++i) buffer[i] = increasing ? 1.0 + i : 3.0 - i;
    auto axis = torch::from_blob(buffer.data(), {3}, opt);
    if (first_address == nullptr) first_address = axis.data_ptr();
    ASSERT_EQ(axis.data_ptr(), first_address);
    auto out = harp::interpn({torch::tensor({1.5}, opt)}, {axis}, lookup);
    EXPECT_NEAR(out[0][0].item<double>(), increasing ? 15.0 : 25.0, 1e-12);
  }
}

TEST(TestInterpolation, testCachesFollowInPlaceMutation) {
  // interpn() does not require immutable inputs, so both caches must notice
  // when a tensor they have seen is written in place: the axis direction may
  // flip, and an on-nodes query may move off the nodes.
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);

  auto axis = torch::tensor({1.0, 2.0, 3.0}, opt);
  auto lookup = torch::tensor({{10.0}, {20.0}, {30.0}}, opt);
  auto query = torch::tensor({1.5}, opt);
  EXPECT_NEAR(harp::interpn({query}, {axis}, lookup)[0][0].item<double>(), 15.0,
              1e-12);
  axis.copy_(axis.flip(0));  // now {3, 2, 1}, decreasing
  EXPECT_NEAR(harp::interpn({query}, {axis}, lookup)[0][0].item<double>(), 25.0,
              1e-12);
  // Writing through a view bumps the same counter.
  axis.slice(0, 0, 3).copy_(torch::tensor({1.0, 2.0, 3.0}, opt));
  EXPECT_NEAR(harp::interpn({query}, {axis}, lookup)[0][0].item<double>(), 15.0,
              1e-12);

  const int nwave = 4;
  auto kwave = torch::linspace(100.0, 400.0, nwave, opt);
  auto table = torch::randn({nwave, 2}, opt);
  auto wave_query = kwave.clone().unsqueeze(-1);
  auto run = [&](torch::Tensor const& q) {
    return harp::interpn({q}, {kwave}, table);
  };
  EXPECT_TRUE(torch::equal(run(wave_query), run(wave_query.clone())));
  wave_query.add_(10.0);  // off the nodes now
  EXPECT_TRUE(torch::equal(run(wave_query), run(wave_query.clone())));
  wave_query.sub_(10.0);  // and back on them
  EXPECT_TRUE(torch::equal(run(wave_query), run(wave_query.clone())));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
