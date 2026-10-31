// C/C++
#include <cmath>

// external
#include <gtest/gtest.h>

// harp
#include <harp/radiation/bbflux.hpp>

// The checks below are independent of the units and normalisation of the
// flux: they test the Planck spectral shape, the T^4 law of the full band and
// the additivity of band integrals.
//
// The shape is checked twice: exactly (1e-9) with the second radiation constant
// each function uses (c2 = 1.4388 cm K for wavenumbers; h c / k with c = 3.0e8
// m/s for wavelengths), and within 1% with the CODATA value, which those
// constants differ from by 1.6e-5 and 6.9e-4.
static double const c2_codata = 1.438776877;  // [cm K]
static double const c2_wavenumber = 1.4388;   // bbflux_wavenumber(wave, ...)
static double const c2_wavelength =
    6.62607015e-34 * 3.0e8 / 1.380649e-23 * 100.0;  // bbflux_wavelength

// Planck shape in wavenumber: F(nu) ~ nu^3 / (exp(c2 nu / T) - 1)
static double planck_nu_shape(double nu, double T, double c2) {
  return nu * nu * nu / std::expm1(c2 * nu / T);
}

// Planck shape in wavelength [um]: F(lambda) ~ lambda^-5 / (exp(c2 / lambda /
// T) - 1)
static double planck_lambda_shape(double lam_um, double T, double c2) {
  double lam_cm = lam_um * 1e-4;
  return std::pow(lam_cm, -5) / std::expm1(c2 / (lam_cm * T));
}

TEST(BBFluxTest, Wavenumber1) {
  torch::Tensor wave =
      torch::tensor({100.0, 500.0, 1000.0, 2000.0}, torch::kFloat64);
  double temp = 300.0;
  int ncol = 2;

  auto result = harp::bbflux_wavenumber(wave, temp, ncol);

  ASSERT_EQ(result.sizes(), (std::vector<int64_t>{4, 2}));
  EXPECT_TRUE(torch::equal(result.select(1, 0), result.select(1, 1)));
  EXPECT_TRUE((result > 0).all().item<bool>());
  for (int i = 1; i < 4; ++i) {
    double nu = wave[i].item<double>(), nu0 = wave[0].item<double>();
    double ratio = result[i][0].item<double>() / result[0][0].item<double>();
    double exact = planck_nu_shape(nu, temp, c2_wavenumber) /
                   planck_nu_shape(nu0, temp, c2_wavenumber);
    double codata = planck_nu_shape(nu, temp, c2_codata) /
                    planck_nu_shape(nu0, temp, c2_codata);
    EXPECT_NEAR(ratio, exact, 1e-9 * exact) << "nu = " << nu;
    EXPECT_NEAR(ratio, codata, 1e-2 * codata) << "nu = " << nu;
  }
}

TEST(BBFluxTest, Wavenumber2) {
  torch::Tensor temp = torch::tensor({100.0, 200.0, 300.0}, torch::kFloat64);

  // over (nearly) the whole spectrum the band integral scales as T^4
  auto full = harp::bbflux_wavenumber(1.0, 1.0e5, temp).flatten();
  ASSERT_EQ(full.numel(), 3);
  for (int i = 1; i < 3; ++i) {
    double ratio = full[i].item<double>() / full[0].item<double>();
    double expected = std::pow(temp[i].item<double>() / 100.0, 4);
    EXPECT_NEAR(ratio, expected, 1e-6 * expected);
  }

  // band integrals add up; the split points keep v = c2 nu / T away from 1.5
  auto lo = harp::bbflux_wavenumber(100.0, 800.0, temp).flatten();
  auto hi = harp::bbflux_wavenumber(800.0, 10000.0, temp).flatten();
  auto both = harp::bbflux_wavenumber(100.0, 10000.0, temp).flatten();
  for (int i = 0; i < 3; ++i) {
    double sum = lo[i].item<double>() + hi[i].item<double>();
    EXPECT_NEAR(sum, both[i].item<double>(), 1e-6 * both[i].item<double>());
    EXPECT_GT(lo[i].item<double>(), 0.0);
    EXPECT_GT(hi[i].item<double>(), 0.0);
  }
}

TEST(BBFluxTest, Wavelength) {
  torch::Tensor wave = torch::tensor({5.0, 10.0, 20.0, 50.0}, torch::kFloat64);
  double temp = 300.0;
  int ncol = 2;

  auto result = harp::bbflux_wavelength(wave, temp, ncol);

  ASSERT_EQ(result.sizes(), (std::vector<int64_t>{4, 2}));
  EXPECT_TRUE(torch::equal(result.select(1, 0), result.select(1, 1)));
  EXPECT_TRUE((result > 0).all().item<bool>());
  for (int i = 1; i < 4; ++i) {
    double lam = wave[i].item<double>(), lam0 = wave[0].item<double>();
    double ratio = result[i][0].item<double>() / result[0][0].item<double>();
    double exact = planck_lambda_shape(lam, temp, c2_wavelength) /
                   planck_lambda_shape(lam0, temp, c2_wavelength);
    double codata = planck_lambda_shape(lam, temp, c2_codata) /
                    planck_lambda_shape(lam0, temp, c2_codata);
    EXPECT_NEAR(ratio, exact, 1e-9 * exact) << "lambda = " << lam;
    EXPECT_NEAR(ratio, codata, 1e-2 * codata) << "lambda = " << lam;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
