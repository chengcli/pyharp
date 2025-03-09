// external
#include <gtest/gtest.h>

// harp
#include <radiation/flux_utils.hpp>

TEST(FluxUtilsTest, CalTotalFluxWave) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 3});
  // Create a 1D tensor with random values
  torch::Tensor wave = torch::rand({5});

  // Calculate the total flux using the wave grid
  torch::Tensor total_flux = harp::cal_total_flux(flux, wave, "wave");

  std::cout << "total_flux: " << total_flux << std::endl;

  // Check the shape of the result
  EXPECT_EQ(total_flux.sizes(), std::vector<int64_t>({3}));
}

TEST(FluxUtilsTest, CalTotalFluxWeight) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 3});
  // Create a 1D tensor with random values
  torch::Tensor weight = torch::rand({5});

  // Calculate the total flux using the weight tensor
  torch::Tensor total_flux = harp::cal_total_flux(flux, weight, "weight");

  std::cout << "total_flux: " << total_flux << std::endl;

  // Check the shape of the result
  EXPECT_EQ(total_flux.sizes(), std::vector<int64_t>({3}));
}

TEST(FluxUtilsTest, CalTotalFluxInvalidInput) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 3});
  // Create a 1D tensor with random values
  torch::Tensor wave = torch::rand({5});

  // Check for invalid input
  EXPECT_THROW(harp::cal_total_flux(flux, wave, "invalid"), c10::Error);
}

TEST(FluxUtilsTest, CalNetFlux) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 2});

  // Calculate the net flux
  torch::Tensor net_flux = harp::cal_net_flux(flux);

  std::cout << "net_flux: " << net_flux << std::endl;

  // Check the shape of the result
  EXPECT_EQ(net_flux.sizes(), std::vector<int64_t>({5}));
}

TEST(FluxUtilsTest, CalNetFluxInvalidInput) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 3});

  // Check for invalid input
  EXPECT_THROW(harp::cal_net_flux(flux), c10::Error);
}

TEST(FluxUtilsTest, CalSurfaceFlux) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({4, 5, 2});

  // Calculate the surface flux
  torch::Tensor surface_flux = harp::cal_surface_flux(flux);

  std::cout << "surface_flux: " << surface_flux << std::endl;

  // Check the shape of the result
  EXPECT_EQ(surface_flux.sizes(), std::vector<int64_t>({4}));
}

TEST(FluxUtilsTest, CalSurfaceFluxInvalidInput) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 3});

  // Check for invalid input
  EXPECT_THROW(harp::cal_surface_flux(flux), c10::Error);
}

TEST(FluxUtilsTest, CalTOAFlux) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({4, 5, 2});

  // Calculate the TOA flux
  torch::Tensor toa_flux = harp::cal_toa_flux(flux);

  std::cout << "toa_flux: " << toa_flux << std::endl;

  // Check the shape of the result
  EXPECT_EQ(toa_flux.sizes(), std::vector<int64_t>({4}));
}

TEST(FluxUtilsTest, CalTOAFluxInvalidInput) {
  // Create a 2D tensor with random values
  torch::Tensor flux = torch::rand({5, 3});

  // Check for invalid input
  EXPECT_THROW(harp::cal_toa_flux(flux), c10::Error);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
