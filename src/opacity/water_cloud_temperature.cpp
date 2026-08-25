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
              "Invalid temperature-switch water-cloud species_id: ",
              species_id, " for conc with ", conc.size(2), " species");
  TORCH_CHECK(kwargs.count("temp") > 0,
              "Temperature-switch water cloud requires temp [K]");

  auto const temp = layer_temperature(kwargs.at("temp"), conc);
  TORCH_CHECK(torch::all(torch::isfinite(temp)).item<bool>() &&
                  torch::all(temp > 0.0).item<bool>(),
              "Temperature-switch water-cloud temp must be finite and "
              "positive");

  auto const ice_mask = temp < kFreezingTemperature;
  auto const liquid_mask = ~ice_mask;

  // Each constituent receives condensate only in its selected phase. This is
  // important because the Fu model validates its effective-radius range only
  // where ice water content is nonzero.
  auto ice_conc = conc.clone();
  ice_conc.select(-1, species_id).mul_(ice_mask);
  auto liquid_conc = conc.clone();
  liquid_conc.select(-1, species_id).mul_(liquid_mask);

  auto const ice_result = ice->forward(ice_conc, kwargs);
  auto const liquid_result = liquid->forward(liquid_conc, kwargs);

  // Select the complete property vector rather than adding the two results:
  // SSA and phase moments are defined even when a constituent's extinction is
  // zero, so addition would contaminate the active phase.
  auto const select_ice = ice_mask.unsqueeze(0).unsqueeze(-1);
  return torch::where(select_ice, ice_result, liquid_result);
}

}  // namespace harp
