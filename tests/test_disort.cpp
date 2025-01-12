// external
#include <gtest/gtest.h>

// rtsolver
#include <rtsolver/rtsolver.hpp>

using namespace harp;

TEST(TestAbsorber, Construct) {
  DisortOptions op;

  op.set_header("running disort example");
  op.set_flags(
      "usrtau,usrang,lamber,quiet,intensity_correction,old_intensity_"
      "correction,print-input,print-phase-function");

  op.ds().nlyr = 1;
  op.ds().nstr = 16;
  op.ds().nmom = 16;

  op.ds().nphi = 1;
  op.ds().ntau = 2;
  op.ds().numu = 6;
  op.nwave() = 1;

  Disort disort(op);
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
