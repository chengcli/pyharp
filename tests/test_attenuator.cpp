// external
#include <gtest/gtest.h>

// opacity
#include <opacity/attenuator.hpp>

using namespace harp;

TEST(TestAbsorber, Construct) {
  AttenuatorOptions op;

  op.name("dummy");
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
