// external
#include <gtest/gtest.h>

// opacity
#include <opacity/s8_fuller.hpp>

using namespace harp;

TEST(TestAttenuation, s8_fuller) {
  S8FullerOptions op;
  op.species_id(0);

  S8Fuller s8(op);
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
