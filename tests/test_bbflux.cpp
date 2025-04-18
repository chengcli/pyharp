// external
#include <gtest/gtest.h>

// harp
#include <harp/radiation/bbflux.hpp>

TEST(BBFluxTest, Wavenumber1) {
  // Test with a simple case
  torch::Tensor wave = torch::tensor({1.0, 2.0, 3.0});
  double temp = 300.0;
  int ncol = 2;

  auto result = harp::bbflux_wavenumber(wave, temp, ncol);

  // Check the shape of the result
  ASSERT_EQ(result.sizes(), (std::vector<int64_t>{3, 2}));

  std::cout << "result: " << result << std::endl;
}

TEST(BBFluxTest, Wavenumber2) {
  torch::Tensor temp = torch::tensor({100.0, 200.0, 300.0});

  auto result = harp::bbflux_wavenumber(100.0, 10000.0, temp);
  std::cout << "result: " << result << std::endl;

  result = harp::bbflux_wavenumber(1.0, 100.0, temp);
  std::cout << "result: " << result << std::endl;

  result = harp::bbflux_wavenumber(10000.0, 1000000.0, temp);
  std::cout << "result: " << result << std::endl;
}

TEST(BBFluxTest, Wavelength) {
  // Test with a simple case
  torch::Tensor wave = torch::tensor({1.0, 2.0, 3.0});
  double temp = 300.0;
  int ncol = 2;

  auto result = harp::bbflux_wavelength(wave, temp, ncol);

  // Check the shape of the result
  ASSERT_EQ(result.sizes(), (std::vector<int64_t>{3, 2}));

  std::cout << "result: " << result << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
