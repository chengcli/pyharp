// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/constants.h>

#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/rayleigh.hpp>

namespace harp {
extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
}  // namespace harp

namespace {

double h2_cross_section_m2_per_mol(double wavenumber_cm1) {
  double const wavelength_angstrom = 1.0e8 / wavenumber_cm1;
  double const sigma_cm2_per_molecule =
      8.14e-13 / std::pow(wavelength_angstrom, 4) +
      1.28e-6 / std::pow(wavelength_angstrom, 6) +
      1.61 / std::pow(wavelength_angstrom, 8);
  return sigma_cm2_per_molecule * harp::constants::Avogadro * 1.0e-4;
}

harp::OpacityOptions rayleigh_options(std::vector<int> species_ids,
                                      int nmom = 4) {
  auto options = harp::OpacityOptionsImpl::create();
  options->type("rayleigh").species_ids(std::move(species_ids)).nmom(nmom);
  return options;
}

}  // namespace

TEST(TestRayleigh, ComputesMixtureAttenuationAndPhaseMoments) {
  harp::species_names = {"H2", "He", "H2O", "CH4", "N2", "CO2", "NH3"};
  harp::species_weights = {2.01588e-3,  4.002602e-3, 18.01528e-3,
                           16.04246e-3, 28.0134e-3,  44.0095e-3,  17.03052e-3};

  harp::Rayleigh rayleigh(rayleigh_options({0, 1, 2, 3, 4, 5, 6}));
  auto conc = torch::ones({1, 1, 7}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavenumber"] = torch::tensor({20000.0}, torch::kFloat64);

  auto result = rayleigh->forward(conc, atm);
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({1, 1, 1, 6}));

  double const scale_sum = 1.0 + 0.0641 + 3.3690 + 10.1509 + 4.6035 + 10.5611 + 7.3427;
  double const expected = h2_cross_section_m2_per_mol(20000.0) * scale_sum;
  EXPECT_NEAR(result[0][0][0][0].item<double>(), expected, expected * 1.0e-12);
  EXPECT_DOUBLE_EQ(result[0][0][0][1].item<double>(), 1.0);
  EXPECT_DOUBLE_EQ(result[0][0][0][2].item<double>(), 0.0);
  EXPECT_DOUBLE_EQ(result[0][0][0][3].item<double>(), 0.1);
  EXPECT_DOUBLE_EQ(result[0][0][0][4].item<double>(), 0.0);
  EXPECT_DOUBLE_EQ(result[0][0][0][5].item<double>(), 0.0);
}

TEST(TestRayleigh, WavelengthAndWavenumberInputsAgree) {
  harp::species_names = {"H2"};
  harp::species_weights = {2.01588e-3};
  harp::Rayleigh rayleigh(rayleigh_options({0}, 2));
  auto conc = torch::tensor({{{2.0}}}, torch::kFloat64);

  std::map<std::string, torch::Tensor> by_wavenumber;
  by_wavenumber["wavenumber"] = torch::tensor({20000.0}, torch::kFloat64);
  std::map<std::string, torch::Tensor> by_wavelength;
  by_wavelength["wavelength"] = torch::tensor({0.5}, torch::kFloat64);

  auto first = rayleigh->forward(conc, by_wavenumber);
  auto second = rayleigh->forward(conc, by_wavelength);
  EXPECT_TRUE(torch::allclose(first, second, 1.0e-12, 1.0e-14));
}

TEST(TestRayleigh, RejectsUnsupportedSpeciesAndTooFewMoments) {
  harp::species_names = {"H2S"};
  harp::species_weights = {34.08088e-3};

  EXPECT_THROW({ harp::Rayleigh rayleigh(rayleigh_options({0})); }, c10::Error);

  harp::species_names = {"H2"};
  harp::species_weights = {2.01588e-3};
  EXPECT_THROW(
      { harp::Rayleigh rayleigh(rayleigh_options({0}, 1)); }, c10::Error);
}
