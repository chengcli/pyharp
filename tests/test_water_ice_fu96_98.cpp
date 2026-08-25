// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// C/C++
#include <cmath>

// harp
#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/water_ice_fu96_98.hpp>

namespace harp {
extern std::vector<double> species_weights;
}  // namespace harp

namespace {

harp::OpacityOptions fu_options(int nmom = 3) {
  auto options = harp::OpacityOptionsImpl::create();
  options->type("water-ice-fu96-98").species_ids({0}).nmom(nmom);
  return options;
}

double re_from_dge(double dge) { return 3.0 * std::sqrt(3.0) * dge / 8.0; }

}  // namespace

TEST(TestFuWaterIce, UsesFu96AndFu98Bands) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options());
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavenumber"] = torch::tensor({20000.0, 1000.0}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);

  auto result = ice->forward(conc, atm);
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({2, 1, 1, 5}));

  double const iwc_g_m3 = 18.01528;
  // 0.5 um: Fu96 interval [0.48, 0.52).
  double const ext96 = -0.945458e-04 + 0.252061e+01 / 50.0;
  double const coalbedo96 = 0.508447e-06 + 0.273206e-07 * 50.0 +
                            0.496553e-10 * 2500.0 -
                            0.186001e-12 * 125000.0;
  double const g96 = 0.749856 + 0.889161e-3 * 50.0 -
                     0.349578e-6 * 2500.0 - 0.109913e-7 * 125000.0;
  EXPECT_NEAR(result[0][0][0][0].item<double>(), ext96 * iwc_g_m3, 1.0e-12);
  EXPECT_NEAR(result[0][0][0][1].item<double>(), 1.0 - coalbedo96, 1.0e-12);
  EXPECT_NEAR(result[0][0][0][2].item<double>(), g96, 1.0e-12);
  EXPECT_NEAR(result[0][0][0][3].item<double>(), g96 * g96, 1.0e-12);
  EXPECT_NEAR(result[0][0][0][4].item<double>(), g96 * g96 * g96, 1.0e-12);

  // 1000 cm^-1 is exactly the 10 um Fu98 wavelength node.
  double const ext98 =
      -7.627102e-03 + 3.406420 / 50.0 - 17.32583 / 2500.0;
  EXPECT_NEAR(result[1][0][0][0].item<double>(), ext98 * iwc_g_m3, 1.0e-12);
}

TEST(TestFuWaterIce, ContainsCompleteTablesAndInterpolatesFu98) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options(1));
  EXPECT_EQ(ice->fu96_wavelength_edges.numel(), 26);
  EXPECT_EQ(ice->fu96_extinction_coeff.sizes(), torch::IntArrayRef({25, 2}));
  EXPECT_EQ(ice->fu96_coalbedo_coeff.sizes(), torch::IntArrayRef({25, 4}));
  EXPECT_EQ(ice->fu96_asymmetry_coeff.sizes(), torch::IntArrayRef({25, 4}));
  EXPECT_EQ(ice->fu96_delta_coeff.sizes(), torch::IntArrayRef({25, 4}));
  EXPECT_EQ(ice->fu98_wavelength.numel(), 36);
  EXPECT_EQ(ice->fu98_extinction_coeff.sizes(), torch::IntArrayRef({36, 3}));
  EXPECT_EQ(ice->fu98_absorption_coeff.sizes(), torch::IntArrayRef({36, 4}));
  EXPECT_EQ(ice->fu98_asymmetry_coeff.sizes(), torch::IntArrayRef({36, 4}));

  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({10.0, 10.5, 11.0}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  auto result = ice->forward(conc, atm);
  EXPECT_NEAR(result[1][0][0][0].item<double>(),
              0.5 * (result[0][0][0][0].item<double>() +
                     result[2][0][0][0].item<double>()),
              1.0e-12);
  EXPECT_NEAR(result[1][0][0][2].item<double>(),
              0.5 * (result[0][0][0][2].item<double>() +
                     result[2][0][0][2].item<double>()),
              1.0e-12);
}

TEST(TestFuWaterIce, ReturnsZeroOutsideFuSpectralRange) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options(3));
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  // 20--50000 cm^-1 spans 500--0.2 um. Only 100--40000 cm^-1 is
  // supported by the combined Fu96/Fu98 tables.
  atm["wavenumber"] =
      torch::tensor({20.0, 100.0, 40000.0, 50000.0}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  auto result = ice->forward(conc, atm);
  EXPECT_TRUE(torch::all(result[0] == 0.0).item<bool>());
  EXPECT_GT(result[1][0][0][0].item<double>(), 0.0);
  EXPECT_GT(result[2][0][0][0].item<double>(), 0.0);
  EXPECT_TRUE(torch::all(result[3] == 0.0).item<bool>());
}

TEST(TestFuWaterIce, AppliesOptionalFu96DeltaScaling) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options(1));
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> raw;
  raw["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  raw["re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  auto scaled = raw;
  scaled["fu_delta_scale"] = torch::tensor(true);
  auto result_raw = ice->forward(conc, raw);
  auto result_scaled = ice->forward(conc, scaled);
  EXPECT_LT(result_scaled[0][0][0][0].item<double>(),
            result_raw[0][0][0][0].item<double>());
  EXPECT_LT(result_scaled[0][0][0][1].item<double>(),
            result_raw[0][0][0][1].item<double>());
  EXPECT_LT(result_scaled[0][0][0][2].item<double>(),
            result_raw[0][0][0][2].item<double>());
}

TEST(TestFuWaterIce, RequiresExplicitRe) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options(1));
  auto conc = torch::tensor({{{2.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["temp"] = torch::tensor({{233.15}}, torch::kFloat64);
  EXPECT_THROW(ice->forward(conc, atm), c10::Error);
}

TEST(TestFuWaterIce, RejectsOutOfRangeExplicitRe) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options(1));
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(10.0), torch::kFloat64);
  EXPECT_THROW(ice->forward(conc, atm), c10::Error);
}

TEST(TestFuWaterIce, KeepsFu98EndpointFitsPhysical) {
  harp::species_weights = {18.01528e-3};
  harp::FuWaterIce ice(fu_options(1));
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({64.67, 75.0}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(11.0), torch::kFloat64);
  auto result = ice->forward(conc, atm);
  EXPECT_TRUE(torch::all(result.select(-1, 1) >= 0.0).item<bool>());
  EXPECT_TRUE(torch::all(result.select(-1, 1) <= 1.0).item<bool>());
}
