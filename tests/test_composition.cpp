// external
#include <gtest/gtest.h>

// harp
#include <harp/compound.hpp>

TEST(Composition, glucose_formula) {
  harp::Composition result = harp::get_composition("C6H12O6");

  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result.at("C"), 6);
  EXPECT_EQ(result.at("H"), 12);
  EXPECT_EQ(result.at("O"), 6);
}

TEST(Composition, molecular_weight) {
  // 6 x 12.011 + 12 x 1.008 + 6 x 15.999 g/mol = 0.180156 kg/mol
  double weight = harp::get_compound_weight(harp::get_composition("C6H12O6"));
  EXPECT_NEAR(weight, 0.180156, 1e-6);

  // water: 2 x 1.008 + 15.999 g/mol
  EXPECT_NEAR(harp::get_compound_weight(harp::get_composition("H2O")), 0.018015,
              1e-6);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
