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

  op.nwve(10);
  op.ds().nlyr = 1;
  op.ds().nstr = 16;
  op.ds().nmom = 16;

  op.ds().nphi = 1;
  op.ds().ntau = 2;
  op.ds().numu = 6;

  Disort disort(op);

  for (int i = 0; i < 10; ++i) {
    disort->ds(i).bc.umu0 = 0.1;
    disort->ds(i).bc.phi0 = 0.0;
    disort->ds(i).bc.albedo = 0.0;
    disort->ds(i).bc.fluor = 0.0;
    disort->ds(i).bc.fisot = 0.0;

    disort->ds(i).umu[0] = -1.;
    disort->ds(i).umu[1] = -0.5;
    disort->ds(i).umu[2] = -0.1;
    disort->ds(i).umu[3] = 0.1;
    disort->ds(i).umu[4] = 0.5;
    disort->ds(i).umu[5] = 1.;

    disort->ds(i).phi[0] = 0.0;

    disort->ds(i).utau[0] = 0.0;
    disort->ds(i).utau[1] = 0.03125;
  }

  auto prop = torch::zeros({disort->options.nwve(), disort->options.ncol(),
                            disort->ds().nlyr, 2 + disort->ds().nstr},
                           torch::kDouble);

  prop.select(3, index::IAB) = disort->ds().utau[1];
  prop.select(3, index::ISS) = 0.2;
  prop.narrow(3, index::IPM, disort->ds().nstr) = scattering_moments(
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
