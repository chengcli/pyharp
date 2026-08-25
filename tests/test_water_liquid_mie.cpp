// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>

// torch
#include <torch/torch.h>

// harp
#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/water_liquid_mie.hpp>

namespace harp {
extern std::vector<double> species_weights;
}  // namespace harp

namespace {

harp::OpacityOptions mie_options(int nmom = 3) {
  auto options = harp::OpacityOptionsImpl::create();
  options->type("water-liquid-mie").species_ids({0}).nmom(nmom);
  return options;
}

}  // namespace

TEST(TestMieWaterLiquid, MatchesMiepythonHomogeneousSphere) {
  harp::species_weights = {18.01528e-3};
  harp::MieWaterLiquid cloud(mie_options());
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  // radius=1 um and wavelength=pi um give size parameter x=2.
  atm["wavelength"] = torch::tensor({std::acos(-1.0)}, torch::kFloat64);
  atm["re"] = torch::tensor(1.0, torch::kFloat64);
  atm["water_density"] = torch::tensor(1000.0, torch::kFloat64);
  atm["refractive_index_real"] = torch::tensor(1.5, torch::kFloat64);
  atm["refractive_index_imag"] = torch::tensor(0.1, torch::kFloat64);

  auto result = cloud->forward(conc, atm);
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({1, 1, 1, 5}));

  // miepython 3.0.2: efficiencies_mx(1.5-0.1j, 2.0).
  double constexpr qext = 1.9414784337094511;
  double constexpr qsca = 1.2861679736548954;
  double constexpr g = 0.65708302470664659;
  double const expected_extinction =
      18.01528e-3 * 3.0 * qext / (4.0 * 1000.0 * 1.0e-6);
  EXPECT_NEAR(result[0][0][0][0].item<double>(), expected_extinction, 1.e-11);
  EXPECT_NEAR(result[0][0][0][1].item<double>(), qsca / qext, 1.e-13);
  EXPECT_NEAR(result[0][0][0][2].item<double>(), g, 1.e-13);
  EXPECT_NEAR(result[0][0][0][3].item<double>(), g * g, 1.e-13);
  EXPECT_NEAR(result[0][0][0][4].item<double>(), g * g * g, 1.e-13);
}

TEST(TestMieWaterLiquid, MatchesMiepythonAtCloudDropletSize) {
  harp::species_weights = {18.01528e-3};
  harp::MieWaterLiquid cloud(mie_options(2));
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  // A 10 um droplet at 0.5 um has x=40*pi, representative of visible-cloud
  // scattering and much larger than the small-particle test above.
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["re"] = torch::tensor(10.0, torch::kFloat64);
  atm["water_density"] = torch::tensor(1000.0, torch::kFloat64);
  atm["refractive_index_real"] = torch::tensor(1.33, torch::kFloat64);
  atm["refractive_index_imag"] = torch::tensor(0.0, torch::kFloat64);

  auto result = cloud->forward(conc, atm);
  // miepython 3.0.2: efficiencies_mx(1.33+0j, 40*pi).
  double constexpr qext = 2.024790517667344;
  double constexpr g = 0.86534861804349872;
  double const expected_extinction =
      18.01528e-3 * 3.0 * qext / (4.0 * 1000.0 * 10.0e-6);
  EXPECT_NEAR(result[0][0][0][0].item<double>(), expected_extinction, 1.e-11);
  EXPECT_NEAR(result[0][0][0][1].item<double>(), 1.0, 1.e-13);
  EXPECT_NEAR(result[0][0][0][2].item<double>(), g, 1.e-13);
  EXPECT_NEAR(result[0][0][0][3].item<double>(), g * g, 1.e-13);
}

TEST(TestMieWaterLiquid, UsesBuiltInSegelsteinWaterIndices) {
  harp::species_weights = {18.01528e-3};
  harp::MieWaterLiquid cloud(mie_options(2));
  auto conc = torch::tensor({{{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] =
      torch::tensor({0.5, 1.0, 10.0, 100.0}, torch::kFloat64);
  atm["re"] = torch::tensor(10.0, torch::kFloat64);
  auto result = cloud->forward(conc, atm);
  EXPECT_TRUE(torch::all(torch::isfinite(result)).item<bool>());
  EXPECT_TRUE(torch::all(result.select(-1, 0) > 0.0).item<bool>());
  EXPECT_TRUE(torch::all(result.select(-1, 1) >= 0.0).item<bool>());
  EXPECT_TRUE(torch::all(result.select(-1, 1) <= 1.0).item<bool>());
}

TEST(TestMieWaterLiquid, ScalesLinearlyWithLiquidWaterContent) {
  harp::species_weights = {18.01528e-3};
  harp::MieWaterLiquid cloud(mie_options(1));
  auto conc = torch::tensor({{{1.0}}, {{2.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.55}, torch::kFloat64);
  atm["re"] = torch::tensor(10.0, torch::kFloat64);
  auto result = cloud->forward(conc, atm);
  EXPECT_NEAR(result[0][1][0][0].item<double>(),
              2.0 * result[0][0][0][0].item<double>(), 1.e-12);
  EXPECT_NEAR(result[0][1][0][1].item<double>(),
              result[0][0][0][1].item<double>(), 1.e-14);
}

TEST(TestMieWaterLiquid, RequiresRadiusAndCompleteIndexOverride) {
  harp::species_weights = {18.01528e-3};
  harp::MieWaterLiquid cloud(mie_options(1));
  auto conc = torch::tensor({{{1.0}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  EXPECT_THROW(cloud->forward(conc, atm), c10::Error);
  atm["re"] = torch::tensor(10.0, torch::kFloat64);
  atm["refractive_index_real"] = torch::tensor(1.33, torch::kFloat64);
  EXPECT_THROW(cloud->forward(conc, atm), c10::Error);
}
