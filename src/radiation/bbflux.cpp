// C/C++
#include <cfloat>
#include <limits>

// harp
#include "bbflux.hpp"

namespace harp {

namespace {

// Exact SI constants (CODATA 2018): second radiation constant h c / k [cm K]
// and Stefan-Boltzmann constant [W m^-2 K^-4]. The band value must equal the
// Planck integral to 1e-6; the rounded constants used before (1.438786,
// 5.67032e-8) alone moved it by up to 5e-5.
constexpr double kC2 = 1.438776877503933;
constexpr double kSigma = 5.670374419184429e-8;

// First radiation constant for a radiance per wavenumber, 2 h c^2 in
// W m^-2 sr^-1 (cm^-1)^-4: 2 h c^2 [W m^2 sr^-1] times 1e8
constexpr double kC1 = 1.1910429723971884e-8;

// Switch between the two series of the band Planck function, in v = c2 nu / T
constexpr double kVcut = 1.5;

// Terms kept of the exponential series; the first neglected one is below
// 1e-12 of sigma T^4 / pi at the switch
constexpr int kNexp = 16;

//! Fraction of sigma T^4 / pi emitted below v, power series (Bernoulli
//! numbers up to B14, first neglected term 6e-12 at v = 1.5)
torch::Tensor planck_fraction_below(torch::Tensor const& v) {
  const double conc = 15.0 / std::pow(M_PI, 4);
  const double A1 = 1.0 / 3.0;
  const double A2 = -1.0 / 8.0;
  const double A3 = 1.0 / 60.0;
  const double A4 = -1.0 / 5040.0;
  const double A5 = 1.0 / 272160.0;
  const double A6 = -1.0 / 13305600.0;
  const double A7 = 1.0 / 622702080.0;
  const double A8 = -691.0 / 19615115520000.0;
  const double A9 = 1.0 / 1270312243200.0;

  auto vsq = v * v;
  return conc * vsq * v *
         (A1 +
          v * (A2 +
               v * (A3 +
                    vsq *
                        (A4 +
                         vsq * (A5 +
                                vsq * (A6 +
                                       vsq * (A7 + vsq * (A8 + vsq * A9))))))));
}

//! Fraction of sigma T^4 / pi emitted above v, exponential series
torch::Tensor planck_fraction_above(torch::Tensor const& v) {
  const double conc = 15.0 / std::pow(M_PI, 4);

  auto ex = torch::exp(-v);
  auto exm = torch::ones_like(v);
  auto d = torch::zeros_like(v);
  for (int m = 1; m <= kNexp; ++m) {
    auto mv = static_cast<double>(m) * v;
    exm = exm * ex;
    double m4 = static_cast<double>(m) * m * m * m;
    d = d + exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / m4;
  }
  return conc * d;
}

//! Planck radiance integrated between v1 = c2 nu1 / T and v2 = c2 nu2 / T, in
//! units of sigma T^4 / pi
torch::Tensor planck_band_fraction(torch::Tensor const& v1,
                                   torch::Tensor const& v2) {
  auto small1 = v1 < kVcut;
  auto small2 = v2 < kVcut;
  auto p1 =
      torch::where(small1, planck_fraction_below(v1), torch::zeros_like(v1));
  auto p2 =
      torch::where(small2, planck_fraction_below(v2), torch::zeros_like(v2));
  auto d1 = torch::where(v1 > kVcut, planck_fraction_above(v1),
                         torch::zeros_like(v1));
  auto d2 = torch::where(v2 > kVcut, planck_fraction_above(v2),
                         torch::zeros_like(v2));

  return torch::where(small1 & small2, p2 - p1,
                      torch::where(small1, 1.0 - p1 - d2, d1 - d2));
}

//! Planck radiance per wavenumber at wavenumber wn [W m^-2 sr^-1 (cm^-1)^-1]
template <typename W, typename T>
torch::Tensor planck_radiance(W const& wn, T const& temp) {
  return kC1 * wn * wn * wn / torch::expm1(kC2 * wn / temp);
}

}  // namespace

torch::Tensor bbflux_wavenumber(torch::Tensor wave, double temp, int ncol) {
  // Check if wave is a 1D tensor
  TORCH_CHECK(wave.dim() == 1, "wavenumber must be a 1D tensor");

  int nwave = wave.size(0);
  auto result = planck_radiance(wave, temp);
  return result.unsqueeze(-1).expand({nwave, ncol}).contiguous();
}

torch::Tensor bbflux_wavenumber(double wn1, double wn2, torch::Tensor temp) {
  if (wn2 < wn1 || wn1 < 0.0) {
    TORCH_CHECK(false, "bbflux_wavenumber: Invalid wavenumbers");
  }

  TORCH_CHECK(temp.min().item<double>() > 0.0,
              "bbflux_wavenumber: Temperature must be positive");

  // Equal edges: the radiance per wavenumber at that wavenumber
  if (wn1 == wn2) {
    return planck_radiance(torch::full_like(temp, wn1), temp);
  }

  auto ans = planck_band_fraction(kC2 * wn1 / temp, kC2 * wn2 / temp);
  return ans * (kSigma / M_PI) * torch::pow(temp, 4);
}

torch::Tensor bbflux_wavenumber(torch::Tensor wn1, torch::Tensor wn2,
                                torch::Tensor temp) {
  if ((wn2 < wn1).any().item<bool>() || (wn1 < 0.0).any().item<bool>()) {
    TORCH_CHECK(false, "bbflux_wavenumber: Invalid wavenumbers");
  }

  TORCH_CHECK(temp.min().item<double>() > 0.0,
              "bbflux_wavenumber: Temperature must be positive");

  // Equal edges give the radiance per wavenumber at that wavenumber, element
  // by element
  auto band = planck_band_fraction(kC2 * wn1 / temp, kC2 * wn2 / temp) *
              (kSigma / M_PI) * torch::pow(temp, 4);
  auto equal = wn1 == wn2;
  if (!equal.any().item<bool>()) return band;
  return torch::where(equal, planck_radiance(wn1, temp), band);
}

torch::Tensor bbflux_wavelength(torch::Tensor wave, double temp, int ncol) {
  // Check if wave is a 1D tensor
  TORCH_CHECK(wave.dim() == 1, "wavelength must be a 1D tensor");

  // Physical constants
  constexpr double h = 6.62607015e-34;  // Planck's constant (J·s)
  constexpr double c = 3.0e8;           // Speed of light (m/s)
  constexpr double kB = 1.380649e-23;   // Boltzmann constant (J/K)

  // Convert wavelength from micrometers to meters
  torch::Tensor wavelength_m = wave * 1e-6;

  // Compute the exponent: hc / (lambda kB T)
  torch::Tensor exponent = (h * c) / (wavelength_m * kB * temp);

  // Compute Planck's law
  torch::Tensor B_lambda =
      2.0 * h * c * c / (wavelength_m.pow(5) * (exponent.exp() - 1.0));

  // Convert flux to per micrometer
  return (B_lambda * 1e-6)
      .unsqueeze(-1)
      .expand({wave.size(0), ncol})
      .contiguous();
}

}  // namespace harp
