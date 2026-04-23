// external
#include <gtest/gtest.h>

// C/C++
#include <filesystem>

// harp
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_band.hpp>
#include <harp/rtsolver/toon_mckay89.hpp>

// tests
#include "device_testing.hpp"

using namespace harp;

TEST(ToonConfig, from_yaml_reads_toon_options) {
  auto yaml_path =
      std::filesystem::path(__FILE__).parent_path() / "toon_test.yaml";
  auto rad = harp::RadiationOptionsImpl::from_yaml(yaml_path.string());
  ASSERT_EQ(rad->bands().size(), 2u);

  auto const& op = rad->bands().front();

  ASSERT_EQ(op->solver_name(), "toon");
  ASSERT_NE(op->toon(), nullptr);
  EXPECT_EQ(op->toon()->flags(),
            "planck,zenith_correction,hard_surface,delta_eddington_lw");
  EXPECT_TRUE(op->toon()->planck());
  EXPECT_TRUE(op->toon()->zenith_correction());
  EXPECT_EQ(op->toon()->top_emission_flag(), -1);
  EXPECT_TRUE(op->toon()->hard_surface());
  EXPECT_TRUE(op->toon()->delta_eddington_lw());
  EXPECT_EQ(op->toon()->wave_lower(),
            (std::vector<double>{200.0, 200.0, 200.0}));
  EXPECT_EQ(op->toon()->wave_upper(),
            (std::vector<double>{2000.0, 2000.0, 2000.0}));
}

TEST(ToonConfig, radiation_band_registers_solver_module) {
  auto op = harp::RadiationBandOptionsImpl::create();
  op->name("B_toon");
  op->solver_name("toon");
  op->toon(harp::ToonMcKay89OptionsImpl::create());
  op->toon()->flags("planck");
  op->nwave(2);
  op->ncol(1);
  op->nlyr(3);
  op->wavenumber({300.0, 900.0});
  op->weight({600.0, 600.0});
  op->set_wave_lower({0.0, 600.0});
  op->set_wave_upper({600.0, 1200.0});

  harp::RadiationBand band(op);

  EXPECT_NO_THROW({ (void)band->named_modules()["solver"]; });
}

TEST(ToonConfig, planck_flag_controls_thermal_emission) {
  auto wave_lower = std::vector<double>{200.0, 500.0};
  auto wave_upper = std::vector<double>{500.0, 1000.0};

  auto sw_op = harp::ToonMcKay89OptionsImpl::create();
  sw_op->wave_lower(wave_lower);
  sw_op->wave_upper(wave_upper);
  harp::ToonMcKay89 sw_toon(sw_op);

  auto lw_op = harp::ToonMcKay89OptionsImpl::create();
  lw_op->wave_lower(wave_lower);
  lw_op->wave_upper(wave_upper);
  lw_op->flags("planck");
  harp::ToonMcKay89 lw_toon(lw_op);

  auto prop = torch::zeros({2, 1, 3, 3}, torch::kFloat64);
  prop.select(-1, 0).fill_(0.2);
  auto temf = torch::ones({1, 4}, torch::kFloat64) * 300.0;
  std::map<std::string, torch::Tensor> sw_bc;
  std::map<std::string, torch::Tensor> lw_bc;

  auto sw_result = sw_toon(prop, &sw_bc, /*band=*/"", temf);
  auto lw_result = lw_toon(prop, &lw_bc, /*band=*/"", temf);

  EXPECT_TRUE(torch::allclose(sw_result, torch::zeros_like(sw_result)));
  EXPECT_GT(torch::max(torch::abs(lw_result)).item<double>(), 0.0);
}

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
  op->flags("planck");

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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
