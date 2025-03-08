// harp
#include "bbflux.hpp"

namespace harp {

torch::Tensor bbflux_wavenumber(torch::Tensor wave, double temp, int ncol) {
  // Check if wave is a 1D tensor
  TORCH_CHECK(wave.dim() == 1, "wavenumber must be a 1D tensor");

  constexpr double c1 = 1.19144e-5 * 1e-3;
  constexpr double c2 = 1.4388;

  int nwave = wave.size(0);
  auto result = c1 * wave.pow(3) / ((c2 * wave / temp).exp() - 1.);
  return result.unsqueeze(-1).expand({nwave, ncol}).contiguous();
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
