// C/C++
#include <stdexcept>
#include <utility>
#include <vector>

// external
#include <gtest/gtest.h>

// harp
#include <harp/utils/read_stellar_flux.hpp>

// in-repo fixtures: band fluxes, and the band-edge wavelengths after a
// two-character header ("# ")
static char const* kFlux = "stellar_flux_test.txt";
static char const* kWave = "stellar_wavelength_test.txt";

static std::vector<double> const kExpectedFlux = {
    1059434.8118773764, 6035941.569908597, 7520131.630087908,
    8488784.244992392,  5729875.721026806, 1400979.114668621,
    1083717.0198599927, 331262.2156410236, 296581.7003886011,
    44557.440115296864, 4298.3275481968085};

static std::vector<double> const kExpectedWave = {0.260, 0.420, 0.610, 0.850,
                                                  1.320, 2.020, 2.500, 3.500,
                                                  4.400, 8.70,  20.00, 324.68};

TEST(read_stellar_flux, fluxes) {
  auto out = read_stellar_flux(kFlux, kWave);
  EXPECT_EQ(out.first, kExpectedFlux);
}

TEST(read_stellar_flux, wavelengths) {
  auto out = read_stellar_flux(kFlux, kWave);
  EXPECT_EQ(out.second, kExpectedWave);
}

TEST(read_stellar_flux, one_more_edge_than_bands) {
  auto out = read_stellar_flux(kFlux, kWave);
  EXPECT_EQ(out.second.size(), out.first.size() + 1);
}

TEST(read_stellar_flux, missing_file_throws) {
  EXPECT_THROW(read_stellar_flux("no_such_file.txt", kWave),
               std::runtime_error);
  EXPECT_THROW(read_stellar_flux(kFlux, "no_such_file.txt"),
               std::runtime_error);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
