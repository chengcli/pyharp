// external
#include <gtest/gtest.h>

// harp
#include <index.h>

#include <rtsolver/rtsolver.hpp>
#include <utils/scattering_moments.hpp>

using namespace harp;

TEST(TestDisort, isotropic_scattering) {
  DisortOptions op;

  op.header("running disort example");
  op.flags(
      "usrtau,usrang,lamber,quiet,intensity_correction,"
      "old_intensity_correction,print-input,print-phase-function");

  op.ds().nlyr = 1;
  op.ds().nstr = 16;
  op.ds().nmom = 16;

  op.ds().nphi = 1;
  op.ds().ntau = 2;
  op.ds().numu = 6;

  Disort disort(op);

  disort->ds().bc.umu0 = 0.1;
  disort->ds().bc.phi0 = 0.0;
  disort->ds().bc.albedo = 0.0;
  disort->ds().bc.fluor = 0.0;
  disort->ds().bc.fisot = 0.0;

  disort->ds().umu[0] = -1.;
  disort->ds().umu[1] = -0.5;
  disort->ds().umu[2] = -0.1;
  disort->ds().umu[3] = 0.1;
  disort->ds().umu[4] = 0.5;
  disort->ds().umu[5] = 1.;

  disort->ds().phi[0] = 0.0;

  disort->ds().utau[0] = 0.0;
  disort->ds().utau[1] = 0.03125;

  auto prop = torch::zeros({disort->options.nwve(), disort->options.ncol(),
                            disort->ds().nlyr, 3 + disort->ds().nstr},
                           torch::kDouble);
  prop.select(3, index::IAB) = disort->ds().utau[1];
  prop.select(3, index::ISS) = 0.2;
  prop.narrow(3, index::IPM, 1 + disort->ds().nstr) = scattering_moments(
      disort->ds().nstr, PhaseMomentOptions().type(kIsotropic));

  auto ftoa = torch::zeros({disort->options.nwve(), disort->options.ncol()},
                           torch::kDouble);
  ftoa.fill_(M_PI / disort->ds().bc.umu0);

  auto result = disort->forward(prop, ftoa);
  std::cout << "result: " << result << std::endl;
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
