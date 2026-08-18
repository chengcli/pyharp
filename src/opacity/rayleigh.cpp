// disort
#include <disort/index.h>

// harp
#include <harp/constants.h>

#include <harp/utils/strings.hpp>

#include "rayleigh.hpp"

namespace harp {

extern std::vector<std::string> species_names;

namespace {

double species_scale(std::string const& species_name) {
  auto const name = to_lower_copy(species_name);
  if (name == "h2") return 1.0;
  if (name == "he") return 0.0641;  // Pierrehumbert (2010)
  if (name == "h2o") return 3.3690;
  if (name == "ch4") return 10.1509;
  if (name == "n2") return 4.6035;
  if (name == "co2") return 10.5611;
  if (name == "nh3") return 7.3427;

  TORCH_CHECK(false, "Rayleigh opacity does not support species '",
              species_name,
              "'. Supported species: H2, He, H2O, CH4, N2, CO2, NH3");
  return 0.0;
}

}  // namespace

RayleighImpl::RayleighImpl(OpacityOptions const& options_) : options(options_) {
  TORCH_CHECK(options->type().empty() || options->type() == "rayleigh",
              "Mismatch opacity type: ", options->type(),
              " expecting 'rayleigh'");
  TORCH_CHECK(!options->species_ids().empty(),
              "Rayleigh opacity requires at least one species");
  TORCH_CHECK(options->nmom() >= 2,
              "Rayleigh opacity requires nmom >= 2 to represent the P2 ",
              "phase-function moment; got ", options->nmom());
  reset();
}

void RayleighImpl::reset() {
  std::vector<double> scales;
  scales.reserve(options->species_ids().size());
  for (auto const species_id : options->species_ids()) {
    TORCH_CHECK(species_id >= 0 && species_id < species_names.size(),
                "Invalid Rayleigh species_id: ", species_id);
    scales.push_back(species_scale(species_names.at(species_id)));
  }
  species_scales =
      register_buffer("species_scales", torch::tensor(scales, torch::kFloat64));
}

torch::Tensor RayleighImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(conc.dim() == 3,
              "Rayleigh expects conc shape (ncol, nlyr, nspecies); got ",
              conc.sizes());

  torch::Tensor wavenumber;
  if (kwargs.count("wavenumber") > 0) {
    wavenumber = kwargs.at("wavenumber");
  } else if (kwargs.count("wavelength") > 0) {
    auto wavelength = kwargs.at("wavelength");
    TORCH_CHECK(torch::all(torch::isfinite(wavelength)).item<bool>() &&
                    torch::all(wavelength > 0.0).item<bool>(),
                "Rayleigh wavelength must be finite and positive");
    wavenumber = 1.0e4 / wavelength;
  } else {
    TORCH_CHECK(false,
                "Rayleigh requires wavenumber [cm^-1] or wavelength [um]");
  }

  TORCH_CHECK(wavenumber.dim() == 1,
              "Rayleigh expects a 1D spectral grid; got ", wavenumber.sizes());
  TORCH_CHECK(torch::all(torch::isfinite(wavenumber)).item<bool>() &&
                  torch::all(wavenumber > 0.0).item<bool>(),
              "Rayleigh wavenumber must be finite and positive");

  wavenumber = wavenumber.to(conc.options());

  // wavelength [Angstrom] = 1e8 / wavenumber [cm^-1]
  auto const wavelength_angstrom = 1.0e8 / wavenumber;

  // Dalgarno & Williams (1962) H2 cross section [cm^2/molecule].
  auto sigma_m2_per_mol = (8.14e-13 * wavelength_angstrom.pow(-4) +
                           1.28e-6 * wavelength_angstrom.pow(-6) +
                           1.61 * wavelength_angstrom.pow(-8)) *
                          (constants::Avogadro * 1.0e-4);

  auto const ncol = conc.size(0);
  auto const nlyr = conc.size(1);
  auto const nwave = wavenumber.size(0);
  auto attenuation = torch::zeros({nwave, ncol, nlyr}, conc.options());
  auto const scales = species_scales.to(conc.options());

  for (int i = 0; i < options->species_ids().size(); ++i) {
    auto const species_id = options->species_ids().at(i);
    TORCH_CHECK(species_id < conc.size(2),
                "Invalid Rayleigh species_id: ", species_id, " for conc with ",
                conc.size(2), " species");
    attenuation += sigma_m2_per_mol.view({nwave, 1, 1}) * scales[i] *
                   conc.select(-1, species_id).unsqueeze(0);
  }

  auto result =
      torch::zeros({nwave, ncol, nlyr, 2 + options->nmom()}, conc.options());
  result.select(-1, disort::IEX).copy_(attenuation);  // extinction [1/m]

  // Rayleigh scattering is conservative: extinction equals scattering.
  result.select(-1, disort::ISS).fill_(1.0);  // single-scattering albedo

  // DISORT stores moments beta_l in
  // P(cos(theta)) = sum_l (2l+1) beta_l P_l(cos(theta)).
  // For P = 3/4 (1 + cos^2(theta)), beta_1=0 and beta_2=0.1.
  result.select(-1, disort::IPM + 1).fill_(0.1);  // second Legendre moment

  return result;
}

}  // namespace harp
