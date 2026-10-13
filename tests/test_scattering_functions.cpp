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

// Zero requested moments still depend on g: the empty result must stay in the
// graph, so that a backward through it gives a defined zero gradient.
TEST_P(DeviceTest, HenyeyGreensteinEmptyMomentsKeepGraph) {
  auto g = torch::tensor({0.25, -0.5}, torch::device(device).dtype(dtype))
               .requires_grad_(true);
  auto result = harp::henyey_greenstein(0, g);

  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 0}));
  EXPECT_EQ(result.scalar_type(), g.scalar_type());
  EXPECT_EQ(result.device(), g.device());
  ASSERT_TRUE(result.requires_grad()) << "nmom = 0 output is detached from g";

  result.backward(torch::empty_like(result));
  ASSERT_TRUE(g.grad().defined());
  EXPECT_TRUE(torch::equal(g.grad(), torch::zeros_like(g)));
}

// The moments are a new tensor: writing to them must neither fail nor reach g.
TEST_P(DeviceTest, HenyeyGreensteinOutputIsOwnedAndWritable) {
  auto values = torch::tensor({0.25, -0.5}, torch::device(device).dtype(dtype));
  for (int nmom : {0, 1, 3}) {
    auto g = values.clone().requires_grad_(true);
    auto result = harp::henyey_greenstein(nmom, g);

    EXPECT_FALSE(result.is_view()) << "nmom = " << nmom;
    EXPECT_FALSE(result.storage().is_alias_of(g.storage()))
        << "nmom = " << nmom;

    torch::NoGradGuard no_grad;
    EXPECT_NO_THROW(result.fill_(0.125)) << "nmom = " << nmom;
    EXPECT_TRUE(torch::equal(g, values)) << "nmom = " << nmom;
  }
}

// Integer and bool inputs pass the range check only when zero; their moments
// keep the input dtype.
TEST_P(DeviceTest, HenyeyGreensteinKeepsIntegerAndBoolDtypes) {
  for (auto int_dtype : {torch::kBool, torch::kInt8, torch::kUInt8,
                         torch::kInt16, torch::kInt32, torch::kInt64}) {
    auto g = torch::zeros({3}, torch::device(device).dtype(int_dtype));
    for (int nmom : {0, 1, 3}) {
      auto result = harp::henyey_greenstein(nmom, g);
      EXPECT_EQ(result.scalar_type(), int_dtype)
          << int_dtype << ", nmom = " << nmom;
      EXPECT_EQ(result.device(), g.device());
      EXPECT_EQ(result.sizes(), (std::vector<int64_t>{3, nmom}));
      EXPECT_TRUE(torch::all(result == 0).item<bool>())
          << int_dtype << ", nmom = " << nmom;
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
