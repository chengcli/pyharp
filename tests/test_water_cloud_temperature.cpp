// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <string>

// torch
#include <torch/torch.h>

// harp
#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/water_cloud_temperature.hpp>
#include <harp/opacity/water_ice_fu96_98.hpp>
#include <harp/opacity/water_liquid_mie.hpp>

namespace harp {
extern std::vector<double> species_weights;
}  // namespace harp

namespace {

harp::OpacityOptions cloud_options(std::string const& type, int nmom = 2) {
  auto options = harp::OpacityOptionsImpl::create();
  options->type(type).species_ids({0}).nmom(nmom);
  return options;
}

double re_from_dge(double dge) { return 3.0 * std::sqrt(3.0) * dge / 8.0; }

}  // namespace

TEST(TestTemperatureSwitchWaterCloud, SelectsIceBelowAndLiquidAtFreezing) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch"));
  harp::FuWaterIce ice(cloud_options("water-ice-fu96-98"));
  harp::MieWaterLiquid liquid(cloud_options("water-liquid-mie"));

  auto conc = torch::tensor({{{0.01}}, {{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  atm["temp"] = torch::tensor({{273.14}, {273.15}}, torch::kFloat64);

  auto const result = cloud->forward(conc, atm);
  auto const ice_result = ice->forward(conc, atm);
  auto const liquid_result = liquid->forward(conc, atm);

  ASSERT_EQ(result.sizes(), torch::IntArrayRef({1, 2, 1, 4}));
  EXPECT_TRUE(torch::allclose(result[0][0], ice_result[0][0]));
  EXPECT_TRUE(torch::allclose(result[0][1], liquid_result[0][1]));
  EXPECT_FALSE(torch::allclose(result[0][0], liquid_result[0][0]));
  EXPECT_FALSE(torch::allclose(result[0][1], ice_result[0][1]));
}

TEST(TestTemperatureSwitchWaterCloud, RequiresTemperature) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch", 1));
  auto conc = torch::tensor({{{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  EXPECT_THROW(cloud->forward(conc, atm), c10::Error);
}
