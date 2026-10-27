// external
#include <gtest/gtest.h>

// fmt
#include <fmt/format.h>

// harp
#include <harp/radiation/radiation_formatter.hpp>

// rayleigh_test.yaml is the configuration test_rayleigh.py also reads: one
// DISORT band of one wavenumber, 4 streams, with a Rayleigh opacity
TEST(TestYAMLInput, reads_the_bands) {
  auto op = harp::RadiationOptionsImpl::from_yaml("rayleigh_test.yaml");

  ASSERT_EQ(op->bands().size(), 1);
  auto const& band = op->bands()[0];
  EXPECT_EQ(band->name(), "shortwave");
  EXPECT_EQ(band->solver_name(), "disort");
  EXPECT_EQ(band->nstr(), 4);
  EXPECT_EQ(band->nwave(), 1);
  EXPECT_EQ(band->opacities().size(), 1);
  EXPECT_EQ(band->opacities().count("rayleigh"), 1);

  // the formatter prints the band it read
  EXPECT_NE(fmt::format("{}", op).find("shortwave"), std::string::npos);
}

TEST(TestYAMLInput, missing_file_throws) {
  EXPECT_ANY_THROW(harp::RadiationOptionsImpl::from_yaml("no_such_file.yaml"));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
