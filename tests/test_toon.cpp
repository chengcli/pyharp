// external
#include <gtest/gtest.h>

// harp
#include <harp/rtsolver/toon_mckay89.hpp>

// tests
#include "device_testing.hpp"

using namespace harp;

TEST_P(DeviceTest, simple_toon_mckay89) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({200., 500., 1000.});
  op->wave_upper({500., 1000., 2000.});

  op->report(std::cout);
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  int nwave = op->wave_lower().size();
  int nlyr = 10;
  int ncol = 2;
  int nprop = 3;

  double tau = 0.1;
  double fbeam = 1.0;
  double umu0 = 0.5;
  double tem_K = 300.0;

  auto prop = torch::zeros({nwave, ncol, nlyr, nprop},
                           torch::device(device).dtype(dtype));
  prop.select(-1, 0) = tau;
  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = torch::ones({nwave, ncol}, prop.options()) * fbeam;
  bc["umu0"] = torch::ones({ncol}, prop.options()) * umu0;
  bc["albedo"] = torch::zeros({nwave, ncol}, prop.options());

  toon.options.planck(false);
  for (auto [w0, g] : {std::make_pair(0.1, 0.5), std::make_pair(0.5, 0.5),
                       std::make_pair(0.9, 0.5)}) {
    std::cout << "w0 = " << w0 << ", g = " << g << "\n";

    prop.select(-1, 1) = w0;  // single scattering albedo
    prop.select(-1, 2) = g;   // asymmetry parameter

    auto sw_flx = toon(prop, &bc);

    std::cout << "sw_flx_up = " << sw_flx.select(-1, 0) << "\n";
    std::cout << "sw_flx_dn = " << sw_flx.select(-1, 1) << "\n";
  }

  auto temf = torch::ones({ncol, nlyr + 1}, prop.options()) * tem_K;

  toon.options.planck(true);
  for (auto [w0, g] : {std::make_pair(0.1, 0.5), std::make_pair(0.5, 0.5),
                       std::make_pair(0.9, 0.5)}) {
    std::cout << "w0 = " << w0 << ", g = " << g << "\n";

    prop.select(-1, 1) = w0;  // single scattering albedo
    prop.select(-1, 2) = g;   // asymmetry parameter
    auto lw_flx = toon(prop, &bc, /*band=*/"", temf);

    std::cout << "lw_flx_up = " << lw_flx.select(-1, 0) << "\n";
    std::cout << "lw_flx_dn = " << lw_flx.select(-1, 1) << "\n";
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
