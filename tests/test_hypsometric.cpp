// C/C++
#include <cmath>

// external
#include <gtest/gtest.h>

// harp
#include <harp/radiation/calc_dz_hypsometric.hpp>

// dz = T dlnp / (g/R): exact for pressures at the layer boundaries
TEST(Hypsometric, level) {
  auto f64 = torch::kFloat64;
  auto pres = torch::tensor({1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0,
                             300.0, 200.0, 100.0}, f64) *
              100.;
  auto temp = torch::tensor(
      {300.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0}, f64);
  double g_ov_R = 9.80665 / 287.05;

  auto dz = harp::calc_dz_hypsometric(pres, temp, torch::tensor(g_ov_R, f64));

  ASSERT_EQ(dz.sizes(), temp.sizes());
  for (int i = 0; i < 9; ++i) {
    double p0 = pres[i].item<double>(), p1 = pres[i + 1].item<double>();
    double expected = temp[i].item<double>() * std::log(p0 / p1) / g_ov_R;
    EXPECT_NEAR(dz[i].item<double>(), expected, 1e-9 * expected) << "layer " << i;
  }
  // 300 K over ln(1000/900): about 925 m
  EXPECT_NEAR(dz[0].item<double>(), 925.0, 1.0);
}

// Pressures at the layer centres are extrapolated to the boundaries in ln p
// with second order; for geometric (ln p linear) centres the boundaries are
// exact, so every layer has dlnp = ln(ratio).
TEST(Hypsometric, layer) {
  double ratio = 0.8;
  auto pres = 1.0e5 * torch::pow(torch::tensor(ratio, torch::kFloat64),
                                 torch::arange(9, torch::kFloat64));
  auto temp = torch::tensor(
      {300.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0}, torch::kFloat64);
  double g_ov_R = 9.80665 / 287.05;

  auto dz = harp::calc_dz_hypsometric(pres, temp, torch::tensor(g_ov_R, torch::kFloat64));

  ASSERT_EQ(dz.sizes(), temp.sizes());
  for (int i = 0; i < 9; ++i) {
    double expected = temp[i].item<double>() * std::log(1.0 / ratio) / g_ov_R;
    EXPECT_NEAR(dz[i].item<double>(), expected, 1e-9 * expected) << "layer " << i;
  }
}

TEST(Hypsometric, mismatched_sizes_throw) {
  auto pres = torch::ones({5}, torch::kFloat64);
  auto temp = torch::ones({3}, torch::kFloat64);
  EXPECT_ANY_THROW(harp::calc_dz_hypsometric(pres, temp, torch::tensor(1.0)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
