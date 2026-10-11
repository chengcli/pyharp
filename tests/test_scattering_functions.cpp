// external
#include <gtest/gtest.h>

// harp
#include <harp/opacity/scattering_functions.hpp>

// tests
#include "device_testing.hpp"

TEST_P(DeviceTest, HenyeyGreensteinMatchesCumprod) {
  auto g = 1.98 * torch::rand({37, 11, 5}, torch::device(device).dtype(dtype)) -
           0.99;

  for (int nmom = 0; nmom <= 6; ++nmom) {
    auto result = harp::henyey_greenstein(nmom, g);

    auto vec = g.sizes().vec();
    vec.push_back(nmom);
    EXPECT_EQ(result.sizes(), vec) << "nmom = " << nmom;
    if (nmom == 0) continue;

    auto expected = torch::cumprod(g.unsqueeze(-1).expand(vec), -1);
    // Up to g^2 the two are the same product. Beyond that cumprod can round
    // differently: on the CPU it accumulates float32 in double, and a CUDA
    // scan may group a long product differently from repeated
    // multiplication.
    bool bitwise = nmom <= 2 || (dtype == torch::kFloat64 &&
                                 (nmom <= 3 || device.type() == torch::kCPU));
    if (bitwise) {
      EXPECT_TRUE(torch::equal(result, expected)) << "nmom = " << nmom;
    } else {
      EXPECT_TRUE(torch::allclose(result, expected)) << "nmom = " << nmom;
    }
  }
}

TEST_P(DeviceTest, HenyeyGreensteinRejectsBadInput) {
  auto g = torch::full({3}, 0.5, torch::device(device).dtype(dtype));
  EXPECT_THROW(harp::henyey_greenstein(-1, g), c10::Error);

  g[1] = 1.0;
  EXPECT_THROW(harp::henyey_greenstein(2, g), c10::Error);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
