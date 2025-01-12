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

  op.nlyr(1);
  op.nstr(16);
  op.nmom(16);

  op.nphi(1);
  op.ntau(2);
  op.numu(6);

  Disort disort(op);
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
