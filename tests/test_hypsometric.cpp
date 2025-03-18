// external
#include <gtest/gtest.h>

// harp
#include <harp/radiation/calc_dz_hypsometric.hpp>

TEST(Hypsometric, level) {
  // pressure [Pa]
  auto pres = torch::tensor({1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0,
                             300.0, 200.0, 100.0}) *
              100.;
  // temperature [K]
  auto temp = torch::tensor(
      {300.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0});

  // gravity over specific gas constant [K/m]
  auto g_ov_R = torch::tensor(9.80665 / 287.05);

  auto dz = harp::calc_dz_hypsometric(pres, temp, g_ov_R);

  std::cout << "dz = " << dz << std::endl;
}

TEST(Hypsometric, layer) {
  // pressure [Pa]
  auto pres = torch::tensor({1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0,
                             300.0, 200.0}) *
              100.;
  // temperature [K]
  auto temp = torch::tensor(
      {300.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0});

  // gravity over specific gas constant [K/m]
  auto g_ov_R = torch::tensor(9.80665 / 287.05);

  auto dz = harp::calc_dz_hypsometric(pres, temp, g_ov_R);

  std::cout << "dz = " << dz << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
