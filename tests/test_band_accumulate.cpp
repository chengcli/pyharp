// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/radiation/band_accumulate.hpp>

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

// An opacity with the given property count. Extinction is zero in some
// cells, as it is where a species is absent, so the averages divide by zero
// somewhere.
torch::Tensor opacity(int nwave, int ncol, int nlyr, int nprop,
                      torch::TensorOptions const& opt) {
  auto k = torch::rand({nwave, ncol, nlyr, nprop}, opt);
  k.select(-1, 0).mul_(torch::rand({nwave, ncol, nlyr}, opt) > 0.3);
  if (nprop > 2) k.narrow(-1, 2, nprop - 2).sub_(0.5);
  return k;
}

}  // namespace

TEST_P(DeviceTest, AccumulateMatchesTensorPath) {
  torch::manual_seed(0);
  auto opt = torch::TensorOptions().dtype(dtype).device(device);
  const int nwave = 7, ncol = 3, nlyr = 5, nprop = 4;

  auto fused = torch::zeros({nwave, ncol, nlyr, nprop}, opt);
  auto reference = torch::zeros({nwave, ncol, nlyr, nprop}, opt);
  for (int nk : {1, 2, 4, 3}) {
    auto k = opacity(nwave, ncol, nlyr, nk, opt);
    harp::accumulate_opacity(fused, k);
    // A tracked opacity keeps the tensor expressions.
    auto tracked = k.clone().set_requires_grad(true);
    harp::accumulate_opacity(reference, tracked);
    EXPECT_FALSE(fused.requires_grad());
    EXPECT_TRUE(same_bits(fused, reference.detach())) << "nprop = " << nk;
  }
}

TEST_P(DeviceTest, FinishMatchesTensorPath) {
  torch::manual_seed(1);
  auto opt = torch::TensorOptions().dtype(dtype).device(device);
  const int nwave = 7, ncol = 3, nlyr = 5;
  double const cap = dtype == torch::kFloat64 ? 1.0 - 1.0e-12 : 1.0 - 1.0e-6;

  for (int nprop : {1, 2, 4}) {
    auto summed = torch::zeros({nwave, ncol, nlyr, nprop}, opt);
    harp::accumulate_opacity(summed, opacity(nwave, ncol, nlyr, nprop, opt));
    if (nprop > 1) {
      // A conservative cell reaches the albedo cap.
      summed[1][2][3][1].copy_(summed[1][2][3][0]);
    }

    for (auto dz : {torch::rand({nlyr}, opt) * 100.0,
                    torch::rand({ncol, nlyr}, opt) * 100.0}) {
      auto fused = summed.clone();
      harp::finish_band_properties(fused, dz, cap);
      // A tracked thickness keeps the tensor expressions.
      auto reference = summed.clone();
      harp::finish_band_properties(reference,
                                   dz.clone().set_requires_grad(true), cap);
      EXPECT_FALSE(fused.requires_grad());
      EXPECT_TRUE(same_bits(fused, reference.detach()))
          << "nprop = " << nprop << ", dz dims = " << dz.dim();
    }
  }
}

TEST_P(DeviceTest, TrackedOpacityReachesTheGraph) {
  auto opt = torch::TensorOptions().dtype(dtype).device(device);
  auto prop = torch::zeros({2, 1, 3, 4}, opt);
  auto k = opacity(2, 1, 3, 4, opt).set_requires_grad(true);
  harp::accumulate_opacity(prop, k);
  ASSERT_TRUE(prop.requires_grad());
  prop.sum().backward();
  ASSERT_TRUE(k.grad().defined());
  // d(sum of extinction)/d(extinction) is one in every cell.
  EXPECT_TRUE(torch::all(k.grad().select(-1, 0) >= 1.0).item<bool>());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
