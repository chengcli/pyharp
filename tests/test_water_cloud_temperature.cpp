// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <string>

// torch
#include <torch/torch.h>

// disort
#include <disort/index.h>

// harp
#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/water_cloud_temperature.hpp>
#include <harp/opacity/water_ice_fu96_98.hpp>
#include <harp/opacity/water_liquid_mie.hpp>

#include "device_testing.hpp"

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

TEST(TestTemperatureSwitchWaterCloud, UsesPurePhasesOutsideMixedPhaseRange) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch"));
  harp::FuWaterIce ice(cloud_options("water-ice-fu96-98"));
  harp::MieWaterLiquid liquid(cloud_options("water-liquid-mie"));

  auto conc =
      torch::tensor({{{0.01}}, {{0.01}}, {{0.01}}, {{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["ice_re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  atm["liquid_re"] = torch::tensor(10.0, torch::kFloat64);
  atm["temp"] =
      torch::tensor({{253.14}, {253.15}, {273.15}, {273.16}}, torch::kFloat64);

  auto const result = cloud->forward(conc, atm);
  auto ice_atm = atm;
  ice_atm["re"] = atm["ice_re"];
  auto liquid_atm = atm;
  liquid_atm["re"] = atm["liquid_re"];
  auto const ice_result = ice->forward(conc, ice_atm);
  auto const liquid_result = liquid->forward(conc, liquid_atm);

  ASSERT_EQ(result.sizes(), torch::IntArrayRef({1, 4, 1, 4}));
  EXPECT_TRUE(torch::allclose(result[0][0], ice_result[0][0]));
  EXPECT_TRUE(torch::allclose(result[0][1], ice_result[0][1]));
  EXPECT_TRUE(torch::allclose(result[0][2], liquid_result[0][2]));
  EXPECT_TRUE(torch::allclose(result[0][3], liquid_result[0][3]));
}

TEST(TestTemperatureSwitchWaterCloud, MixesOpticalPropertiesByPhaseFraction) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch"));
  harp::FuWaterIce ice(cloud_options("water-ice-fu96-98"));
  harp::MieWaterLiquid liquid(cloud_options("water-liquid-mie"));

  auto conc = torch::tensor({{{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["ice_re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  atm["liquid_re"] = torch::tensor(10.0, torch::kFloat64);
  atm["temp"] = torch::tensor({{263.15}}, torch::kFloat64);

  auto const result = cloud->forward(conc, atm);
  auto const half_conc = 0.5 * conc;
  auto ice_atm = atm;
  ice_atm["re"] = atm["ice_re"];
  auto liquid_atm = atm;
  liquid_atm["re"] = atm["liquid_re"];
  auto const ice_result = ice->forward(half_conc, ice_atm);
  auto const liquid_result = liquid->forward(half_conc, liquid_atm);

  auto const ice_extinction = ice_result.select(-1, disort::IEX);
  auto const liquid_extinction = liquid_result.select(-1, disort::IEX);
  auto const extinction = ice_extinction + liquid_extinction;
  auto const ice_scattering =
      ice_extinction * ice_result.select(-1, disort::ISS);
  auto const liquid_scattering =
      liquid_extinction * liquid_result.select(-1, disort::ISS);
  auto const scattering = ice_scattering + liquid_scattering;

  EXPECT_TRUE(torch::allclose(result.select(-1, disort::IEX), extinction));
  EXPECT_TRUE(
      torch::allclose(result.select(-1, disort::ISS), scattering / extinction));
  auto const expected_moments =
      (ice_result.narrow(-1, disort::IPM, 2) * ice_scattering.unsqueeze(-1) +
       liquid_result.narrow(-1, disort::IPM, 2) *
           liquid_scattering.unsqueeze(-1)) /
      scattering.unsqueeze(-1);
  EXPECT_TRUE(
      torch::allclose(result.narrow(-1, disort::IPM, 2), expected_moments));
}

TEST(TestTemperatureSwitchWaterCloud, RequiresTemperature) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch", 1));
  auto conc = torch::tensor({{{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["ice_re"] = torch::tensor(re_from_dge(50.0), torch::kFloat64);
  atm["liquid_re"] = torch::tensor(10.0, torch::kFloat64);
  EXPECT_THROW(cloud->forward(conc, atm), c10::Error);
}

TEST(TestTemperatureSwitchWaterCloud, UsesDefaultsWhenRadiiAreAbsent) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch", 1));
  auto conc = torch::tensor({{{0.01}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, torch::kFloat64);
  atm["temp"] = torch::tensor({{263.15}}, torch::kFloat64);

  auto const default_result = cloud->forward(conc, atm);
  EXPECT_TRUE(torch::all(torch::isfinite(default_result)).item<bool>());

  atm["liquid_re"] = torch::tensor(14.0, torch::kFloat64);
  auto const explicit_result = cloud->forward(conc, atm);
  EXPECT_TRUE(torch::allclose(default_result, explicit_result));
}

TEST_P(DeviceTest, TemperatureSwitchWaterCloudMatchesCpuReference) {
  harp::species_weights = {18.01528e-3};
  harp::TemperatureSwitchWaterCloud cloud(
      cloud_options("water-cloud-temperature-switch", 2));

  auto tensor_options = torch::TensorOptions().dtype(dtype).device(device);
  auto conc = torch::full({1, 2, 1}, 0.01, tensor_options);
  std::map<std::string, torch::Tensor> atm;
  atm["wavelength"] = torch::tensor({0.5}, tensor_options);
  atm["ice_re"] = torch::full({1, 2}, re_from_dge(50.0), tensor_options);
  atm["liquid_re"] = torch::full({1, 2}, 10.0, tensor_options);
  atm["temp"] = torch::tensor({{260.0, 280.0}}, tensor_options);

  auto result = cloud->forward(conc, atm);
  EXPECT_EQ(result.device().type(), device.type());
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({1, 1, 2, 4}));
  EXPECT_TRUE(torch::all(torch::isfinite(result.cpu())).item<bool>());

  harp::TemperatureSwitchWaterCloud cpu_cloud(
      cloud_options("water-cloud-temperature-switch", 2));
  auto cpu_conc = conc.cpu();
  std::map<std::string, torch::Tensor> cpu_atm;
  for (auto const& item : atm) {
    cpu_atm[item.first] = item.second.cpu();
  }
  auto expected = cpu_cloud->forward(cpu_conc, cpu_atm);
  EXPECT_TRUE(torch::allclose(result.cpu(), expected, 1.0e-5, 1.0e-7));
}
