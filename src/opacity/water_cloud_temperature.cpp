// disort
#include <disort/index.h>

// harp
#include "water_cloud_temperature.hpp"

namespace harp {

namespace {

torch::Tensor layer_temperature(torch::Tensor value,
                                torch::Tensor const& conc) {
  value = value.to(conc.options());
  if (value.dim() == 0) {
    return value.expand({conc.size(0), conc.size(1)});
  }
  TORCH_CHECK(value.dim() == 2 && value.size(0) == conc.size(0) &&
                  value.size(1) == conc.size(1),
              "Temperature-switch water cloud expects temp shape "
              "(ncol, nlyr) or a scalar; got ",
              value.sizes());
  return value;
}

torch::Tensor divide_where_positive(torch::Tensor numerator,
                                    torch::Tensor denominator) {
  while (denominator.dim() < numerator.dim()) {
    denominator = denominator.unsqueeze(-1);
  }
  auto const positive = denominator > 0.0;
  auto const safe_denominator =
      torch::where(positive, denominator, torch::ones_like(denominator));
  return torch::where(positive, numerator / safe_denominator,
                      torch::zeros_like(numerator));
}

}  // namespace

TemperatureSwitchWaterCloudImpl::TemperatureSwitchWaterCloudImpl(
    OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->type().empty() ||
                  options->type() == "water-cloud-temperature-switch",
              "Mismatch opacity type: ", options->type(),
              " expecting 'water-cloud-temperature-switch'");
  TORCH_CHECK(options->species_ids().size() == 1,
              "Temperature-switch water cloud requires exactly one water "
              "condensate species");
  TORCH_CHECK(options->species_ids()[0] >= 0,
              "Invalid temperature-switch water-cloud species_id: ",
              options->species_ids()[0]);
  TORCH_CHECK(options->nmom() >= 1,
              "Temperature-switch water cloud requires nmom >= 1; got ",
              options->nmom());
  reset();
}

void TemperatureSwitchWaterCloudImpl::reset() {
  auto ice_options = options->clone();
  ice_options->type("water-ice-fu96-98");
  ice = register_module("ice", FuWaterIce(ice_options));

  auto liquid_options = options->clone();
  liquid_options->type("water-liquid-mie");
  liquid = register_module("liquid", MieWaterLiquid(liquid_options));
}

torch::Tensor TemperatureSwitchWaterCloudImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(conc.dim() == 3,
              "Temperature-switch water cloud expects conc shape "
              "(ncol, nlyr, nspecies); got ",
              conc.sizes());
  auto const species_id = options->species_ids()[0];
  TORCH_CHECK(species_id < conc.size(2),
              "Invalid temperature-switch water-cloud species_id: ", species_id,
              " for conc with ", conc.size(2), " species");
  TORCH_CHECK(kwargs.count("temp") > 0,
              "Temperature-switch water cloud requires temp [K]");
  TORCH_CHECK(kwargs.count("ice_re") > 0,
              "Temperature-switch water cloud requires ice_re [um]");
  TORCH_CHECK(kwargs.count("liquid_re") > 0,
              "Temperature-switch water cloud requires liquid_re [um]");

  auto const temp = layer_temperature(kwargs.at("temp"), conc);
  TORCH_CHECK(torch::all(torch::isfinite(temp)).item<bool>() &&
                  torch::all(temp > 0.0).item<bool>(),
              "Temperature-switch water-cloud temp must be finite and "
              "positive");

  // Khairoutdinov and Randall (2003), Eq. A13: omega_n is the liquid fraction
  // of nonprecipitating condensate. Ice first appears below 273.15 K, the two
  // phases coexist across a 20 K interval, and all condensate is ice at or
  // below 253.15 K.
  auto const liquid_fraction = ((temp - kIceOnlyTemperature) /
                                (kFreezingTemperature - kIceOnlyTemperature))
                                   .clamp(0.0, 1.0);
  auto const ice_fraction = 1.0 - liquid_fraction;

  // Partition only the configured condensate species. This also ensures that
  // the Fu model validates effective radius only where ice is present.
  auto ice_conc = conc.clone();
  ice_conc.select(-1, species_id).mul_(ice_fraction);
  auto liquid_conc = conc.clone();
  liquid_conc.select(-1, species_id).mul_(liquid_fraction);

  // The constituent models both use the generic key `re`; give each model its
  // own effective radius without moving or copying the underlying tensor.
  auto ice_kwargs = kwargs;
  ice_kwargs["re"] = kwargs.at("ice_re");
  auto liquid_kwargs = kwargs;
  liquid_kwargs["re"] = kwargs.at("liquid_re");
  auto const ice_result = ice->forward(ice_conc, ice_kwargs);
  auto const liquid_result = liquid->forward(liquid_conc, liquid_kwargs);

  // Combine the two populations as independent optical constituents.
  // Extinction adds directly; SSA is extinction weighted; phase moments are
  // scattering weighted.
  auto const ice_extinction = ice_result.select(-1, disort::IEX);
  auto const liquid_extinction = liquid_result.select(-1, disort::IEX);
  auto const extinction = ice_extinction + liquid_extinction;
  auto const ice_scattering =
      ice_extinction * ice_result.select(-1, disort::ISS);
  auto const liquid_scattering =
      liquid_extinction * liquid_result.select(-1, disort::ISS);
  auto const scattering = ice_scattering + liquid_scattering;

  auto result = torch::zeros_like(ice_result);
  result.select(-1, disort::IEX).copy_(extinction);
  result.select(-1, disort::ISS)
      .copy_(divide_where_positive(scattering, extinction));

  auto const nmom = options->nmom();
  auto const weighted_moments =
      ice_result.narrow(-1, disort::IPM, nmom) * ice_scattering.unsqueeze(-1) +
      liquid_result.narrow(-1, disort::IPM, nmom) *
          liquid_scattering.unsqueeze(-1);
  result.narrow(-1, disort::IPM, nmom)
      .copy_(divide_where_positive(weighted_moments, scattering));
  return result;
}

}  // namespace harp
