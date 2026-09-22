// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/constants.h>

#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/rayleigh.hpp>
#include <harp/radiation/radiation_band.hpp>
#include <harp/rtsolver/toon_mckay89.hpp>

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
  harp::species_weights = {2.01588e-3, 4.002602e-3, 18.01528e-3, 16.04246e-3,
                           28.0134e-3, 44.0095e-3,  17.03052e-3};

  harp::Rayleigh rayleigh(rayleigh_options({0, 1, 2, 3, 4, 5, 6}));
  auto conc = torch::ones({1, 1, 7}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["wavenumber"] = torch::tensor({20000.0}, torch::kFloat64);

  auto result = rayleigh->forward(conc, atm);
  ASSERT_EQ(result.sizes(), torch::IntArrayRef({1, 1, 1, 6}));

  double const scale_sum =
      1.0 + 0.0641 + 3.3690 + 10.1509 + 4.6035 + 10.5611 + 7.3427;
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

TEST(TestRayleigh, RevalidatesWhenGridChangesAcrossCalls) {
  harp::species_names = {"H2"};
  harp::species_weights = {2.01588e-3};
  harp::Rayleigh rayleigh(rayleigh_options({0}, 2));
  auto conc = torch::tensor({{{2.0}}}, torch::kFloat64);

  std::map<std::string, torch::Tensor> valid;
  valid["wavenumber"] = torch::tensor({20000.0}, torch::kFloat64);
  auto first = rayleigh->forward(conc, valid);
  auto second = rayleigh->forward(conc, valid);
  EXPECT_TRUE(torch::allclose(first, second, 1.0e-12, 1.0e-14));

  std::map<std::string, torch::Tensor> invalid;
  invalid["wavenumber"] = torch::tensor({-20000.0}, torch::kFloat64);
  EXPECT_THROW({ rayleigh->forward(conc, invalid); }, c10::Error);
}

TEST(TestRayleigh, RevalidatesStridedViewSharingTheFirstElement) {
  // The validation cache identifies a grid by its metadata, never its values.
  // Two views of one buffer can share the first element, shape, dtype, and
  // device while reading different values, so strides must be part of the
  // identity or the invalid view below would ride on the valid view's pass.
  harp::species_names = {"H2"};
  harp::species_weights = {2.01588e-3};
  harp::Rayleigh rayleigh(rayleigh_options({0}, 2));
  auto conc = torch::tensor({{{2.0}}}, torch::kFloat64);

  auto buffer = torch::tensor({20000.0, -1.0, 30000.0, -1.0}, torch::kFloat64);
  auto valid_view = buffer.slice(0, 0, 4, 2);    // {20000, 30000}
  auto invalid_view = buffer.slice(0, 0, 2, 1);  // {20000, -1}
  ASSERT_EQ(valid_view.data_ptr(), invalid_view.data_ptr());
  ASSERT_EQ(valid_view.sizes(), invalid_view.sizes());

  std::map<std::string, torch::Tensor> valid;
  valid["wavenumber"] = valid_view;
  rayleigh->forward(conc, valid);
  rayleigh->forward(conc, valid);

  std::map<std::string, torch::Tensor> invalid;
  invalid["wavenumber"] = invalid_view;
  EXPECT_THROW({ rayleigh->forward(conc, invalid); }, c10::Error);
}

TEST(TestRayleigh, RevalidatesAfterInPlaceMutation) {
  // The validation cache keys on tensor identity, so an in-place write to a
  // grid that passed validation must not ride on that pass.
  harp::species_names = {"H2"};
  harp::species_weights = {2.01588e-3};
  harp::Rayleigh rayleigh(rayleigh_options({0}, 2));
  auto conc = torch::tensor({{{2.0}}}, torch::kFloat64);

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["wavenumber"] = torch::tensor({20000.0, 30000.0}, torch::kFloat64);
  rayleigh->forward(conc, kwargs);
  kwargs["wavenumber"].fill_(-1.0);
  EXPECT_THROW({ rayleigh->forward(conc, kwargs); }, c10::Error);

  std::map<std::string, torch::Tensor> by_wavelength;
  by_wavelength["wavelength"] = torch::tensor({0.5, 0.3}, torch::kFloat64);
  rayleigh->forward(conc, by_wavelength);
  by_wavelength["wavelength"].select(0, 1).fill_(-0.3);
  EXPECT_THROW({ rayleigh->forward(conc, by_wavelength); }, c10::Error);
}

TEST(TestRayleigh, BandRebuildsSpectralGridMutatedThroughKwargs) {
  // RadiationBand hands its cached wavenumber and wavelength tensors to the
  // caller through kwargs on every step. A caller that writes to one of them
  // in place must not poison later steps: the band has to notice and rebuild
  // both, so the opacities keep seeing a valid, reciprocal grid.
  harp::species_names = {"H2"};
  harp::species_weights = {2.01588e-3};

  auto band_options = harp::RadiationBandOptionsImpl::create();
  band_options->name("shortwave")
      .solver_name("toon")
      .toon(harp::ToonMcKay89OptionsImpl::create())
      .nwave(2)
      .ncol(1)
      .nlyr(2)
      .wavenumber({20000.0, 30000.0})
      .weight({1.0, 1.0});
  band_options->set_wave_lower({19999.5, 29999.5});
  band_options->set_wave_upper({20000.5, 30000.5});
  band_options->opacities()["rayleigh"] = rayleigh_options({0}, 4);
  harp::RadiationBand band(band_options);

  auto conc = torch::tensor({{{2.0}, {3.0}}}, torch::kFloat64);
  auto dz = torch::ones({2}, torch::kFloat64);
  std::map<std::string, torch::Tensor> bc;
  std::map<std::string, torch::Tensor> kwargs;

  auto first = band->forward(conc, dz, &bc, &kwargs).clone();
  auto first_grid = kwargs.at("wavenumber");
  auto first_prop = band->prop.clone();

  // Same grid tensor handed out again on an untouched second step.
  band->forward(conc, dz, &bc, &kwargs);
  EXPECT_EQ(kwargs.at("wavenumber").data_ptr(), first_grid.data_ptr());

  // Corrupt the caller-visible alias; the next step must not use it.
  kwargs.at("wavenumber").fill_(-1.0);
  auto third = band->forward(conc, dz, &bc, &kwargs);
  EXPECT_TRUE(torch::equal(kwargs.at("wavenumber"),
                           torch::tensor({20000.0, 30000.0}, torch::kFloat64)));
  EXPECT_TRUE(torch::allclose(kwargs.at("wavelength"),
                              1.0e4 / kwargs.at("wavenumber")));
  EXPECT_TRUE(torch::equal(band->prop, first_prop));
  EXPECT_TRUE(torch::equal(third, first));

  // The other alias is guarded the same way.
  kwargs.at("wavelength").mul_(2.0);
  band->forward(conc, dz, &bc, &kwargs);
  EXPECT_TRUE(torch::allclose(kwargs.at("wavelength"),
                              1.0e4 / kwargs.at("wavenumber")));
  EXPECT_TRUE(torch::equal(band->prop, first_prop));
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
