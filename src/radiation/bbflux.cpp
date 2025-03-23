// C/C++
#include <cfloat>
#include <limits>

// harp
#include "bbflux.hpp"

namespace harp {

torch::Tensor bbflux_wavenumber(torch::Tensor wave, double temp, int ncol) {
  // Check if wave is a 1D tensor
  TORCH_CHECK(wave.dim() == 1, "wavenumber must be a 1D tensor");

  // Physical constants
  constexpr double c1 = 1.19144e-5 * 1e-3;
  constexpr double c2 = 1.4388;

  int nwave = wave.size(0);
  auto result = c1 * wave.pow(3) / ((c2 * wave / temp).exp() - 1.);
  return result.unsqueeze(-1).expand({nwave, ncol}).contiguous();
}

torch::Tensor bbflux_wavenumber(double wn1, double wn2, torch::Tensor temp) {
  if (wn2 < wn1 || wn1 < 0.0) {
    TORCH_CHECK(false, "bbflux_wavenumber: Invalid wavenumbers");
  }

  TORCH_CHECK(temp.min().item<double>() > 0.0,
              "bbflux_wavenumber: Temperature must be positive");

  const double C2 = 1.438786;       // h * c / k in units cm * K
  const double SIGMA = 5.67032e-8;  // Stefan-Boltzmann constant in W/m²K⁴
  const double VCUT = 1.5;
  const double sigdpi = SIGMA / M_PI;
  const double vmax = std::log(DBL_MAX);
  const double conc = 15.0 / std::pow(M_PI, 4);  // Now computed at runtime
  const double c1 = 1.1911e-18;  // h * c^2, in units W/(m² * sr * cm⁻⁴)
  const double A1 = 1.0 / 3.0;
  const double A2 = -1.0 / 8.0;
  const double A3 = 1.0 / 60.0;
  const double A4 = -1.0 / 5040.0;
  const double A5 = 1.0 / 272160.0;
  const double A6 = -1.0 / 13305600.0;

  // Handle the case where wn1 == wn2
  if (wn1 == wn2) {
    double wn = wn1;
    auto arg = torch::exp(-C2 * wn / temp);
    return c1 * std::pow(wn, 3) * arg / (1.0 - arg);
  }

  torch::Tensor v[2] = {C2 * wn1 / temp, C2 * wn2 / temp};
  torch::Tensor smallv = torch::zeros_like(temp);
  torch::Tensor p[2];
  torch::Tensor d[2];

  // Handle different cases for wavenumbers
  for (int i = 0; i <= 1; ++i) {
    smallv += torch::where(v[i] < VCUT, torch::ones_like(temp),
                           torch::zeros_like(temp));

    auto vsq = v[i] * v[i];
    p[i] =
        conc * vsq * v[i] *
        (A1 + v[i] * (A2 + v[i] * (A3 + vsq * (A4 + vsq * (A5 + vsq * A6)))));
    p[i] = torch::where(v[i] < VCUT, p[i], torch::zeros_like(temp));

    // Use exponential series expansion
    const double vcp[7] = {10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0};

    auto ex = torch::exp(-v[i]);
    auto exm = torch::ones_like(temp);
    d[i] = torch::zeros_like(temp);

    for (int m = 1; m <= 6; ++m) {
      auto mv = static_cast<double>(m) * v[i];
      exm *= ex;
      d[i] += exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / (m * m);
    }
    d[i] *= conc;

    d[i] = torch::where(v[i] > VCUT, d[i], torch::zeros_like(temp));
  }

  auto ans =
      torch::where(smallv == 2, p[1] - p[0],
                   torch::where(smallv == 1, 1.0 - p[0] - d[1], d[0] - d[1]));

  return ans * sigdpi * torch::pow(temp, 4);
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
