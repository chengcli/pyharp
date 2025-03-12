// external
#include <gtest/gtest.h>

// fmt
#include <fmt/format.h>

// harp
#include <radiation/radiation_formatter.hpp>

TEST(TestYAMLInput, test) {
  // Test the YAML input
  auto op = harp::RadiationOptions::from_yaml("amars-ck.yaml");

  std::cout << "op = " << fmt::format("{}", op) << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
