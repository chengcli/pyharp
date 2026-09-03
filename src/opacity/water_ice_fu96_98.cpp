// C/C++
#include <cstddef>

// disort
#include <disort/index.h>

// harp
#include <harp/utils/find_resource.hpp>
#include <harp/utils/read_data_tensor.hpp>

#include "scattering_functions.hpp"
#include "water_ice_fu96_98.hpp"

namespace harp {

extern std::vector<double> species_weights;

namespace {

// Fu (1996), Table 2: range of the 28 particle-size distributions used to
// develop the solar parameterization.
constexpr double kFu96MinDge = 18.63;  // um
constexpr double kFu96MaxDge = 130.24;
constexpr double kFu98MinDge = 11.0;
constexpr double kFu98MaxDge = 129.6;
// Fu (1996), Eq. (3.12): re = 3 sqrt(3) Dge / 8.
constexpr double kDgeToRe = 0.649519052838329;
constexpr double kReToDge = 1.0 / kDgeToRe;
constexpr double kSafeDge = 50.0;  // um; used only where IWC is zero

torch::Tensor load_cloud_table(std::string const& name,
                               torch::IntArrayRef expected) {
  auto table = read_data_tensor(find_resource("opacity/cloud/" + name));
  if (expected.size() == 1) {
    table = table.flatten();
  }
  TORCH_CHECK(table.sizes() == expected, "Cloud opacity data file ", name,
              " has shape ", table.sizes(), "; expected ", expected);
  return table;
}

torch::Tensor layer_field(torch::Tensor value, torch::Tensor const& conc,
                          char const* name) {
  value = value.to(conc.options());
  if (value.dim() == 0) {
    return value.expand({conc.size(0), conc.size(1)});
  }
  TORCH_CHECK(value.dim() == 2 && value.size(0) == conc.size(0) &&
                  value.size(1) == conc.size(1),
              "Fu water ice expects ", name,
              " shape (ncol, nlyr) or a scalar; got ", value.sizes());
  return value;
}

// Sun and Rikus (1999), with the multiplicative temperature correction from
// Sun (2001). T is in K, IWC is in g/m^3, and the result is Fu's generalized
// effective size Dge in um.
torch::Tensor sun_rikus_dge(torch::Tensor const& temp,
                            torch::Tensor const& iwc) {
  auto const factor = 1.2351 + 0.0105 * (temp - 273.15);
  auto const a = 45.8966 * iwc.pow(0.2214);
  auto const b = 0.7957 * iwc.pow(0.2535);
  return factor * (a + b * (temp - 83.15));
}

torch::Tensor fu96_extinction(torch::Tensor const& coeff,
                              torch::Tensor const& d) {
  return coeff.select(1, 0).view({-1, 1, 1}) +
         coeff.select(1, 1).view({-1, 1, 1}) / d;
}

torch::Tensor fu96_coalbedo(torch::Tensor const& coeff,
                            torch::Tensor const& d) {
  return coeff.select(1, 0).view({-1, 1, 1}) +
         coeff.select(1, 1).view({-1, 1, 1}) * d +
         coeff.select(1, 2).view({-1, 1, 1}) * d.pow(2) +
         coeff.select(1, 3).view({-1, 1, 1}) * d.pow(3);
}

torch::Tensor fu96_asymmetry(torch::Tensor const& coeff,
                             torch::Tensor const& d) {
  return coeff.select(1, 0).view({-1, 1, 1}) +
         coeff.select(1, 1).view({-1, 1, 1}) * d +
         coeff.select(1, 2).view({-1, 1, 1}) * d.pow(2) +
         coeff.select(1, 3).view({-1, 1, 1}) * d.pow(3);
}

// Fu (1996), Eq. (3.9d): forward delta-transmission fraction f_delta.
torch::Tensor fu96_delta_fraction(torch::Tensor const& coeff,
                                  torch::Tensor const& d) {
  return coeff.select(1, 0).view({-1, 1, 1}) +
         coeff.select(1, 1).view({-1, 1, 1}) * d +
         coeff.select(1, 2).view({-1, 1, 1}) * d.pow(2) +
         coeff.select(1, 3).view({-1, 1, 1}) * d.pow(3);
}

torch::Tensor fu98_extinction(torch::Tensor const& coeff,
                              torch::Tensor const& d) {
  return coeff.select(1, 0).view({-1, 1, 1}) +
         coeff.select(1, 1).view({-1, 1, 1}) / d +
         coeff.select(1, 2).view({-1, 1, 1}) / d.pow(2);
}

torch::Tensor fu98_absorption(torch::Tensor const& coeff,
                              torch::Tensor const& d) {
  return (coeff.select(1, 0).view({-1, 1, 1}) +
          coeff.select(1, 1).view({-1, 1, 1}) * d +
          coeff.select(1, 2).view({-1, 1, 1}) * d.pow(2) +
          coeff.select(1, 3).view({-1, 1, 1}) * d.pow(3)) /
         d;
}

torch::Tensor fu98_asymmetry(torch::Tensor const& coeff,
                             torch::Tensor const& d) {
  return coeff.select(1, 0).view({-1, 1, 1}) +
         coeff.select(1, 1).view({-1, 1, 1}) * d +
         coeff.select(1, 2).view({-1, 1, 1}) * d.pow(2) +
         coeff.select(1, 3).view({-1, 1, 1}) * d.pow(3);
}

}  // namespace

FuWaterIceImpl::FuWaterIceImpl(OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->type().empty() || options->type() == "water-ice-fu96-98",
              "Mismatch opacity type: ", options->type(),
              " expecting 'water-ice-fu96-98'");
  TORCH_CHECK(options->species_ids().size() == 1,
              "Fu water ice requires exactly one H2O(s) species");
  TORCH_CHECK(options->species_ids()[0] >= 0,
              "Invalid Fu water-ice species_id: ", options->species_ids()[0]);
  TORCH_CHECK(options->nmom() >= 1, "Fu water ice requires nmom >= 1; got ",
              options->nmom());
  reset();
}

void FuWaterIceImpl::reset() {
  fu96_wavelength_edges =
      register_buffer("fu96_wavelength_edges",
                      load_cloud_table("kFu96WavelengthEdges.txt", {26}));
  fu96_extinction_coeff =
      register_buffer("fu96_extinction_coeff",
                      load_cloud_table("kFu96Extinction.txt", {25, 2}));
  fu96_coalbedo_coeff = register_buffer(
      "fu96_coalbedo_coeff", load_cloud_table("kFu96Coalbedo.txt", {25, 4}));
  fu96_asymmetry_coeff = register_buffer(
      "fu96_asymmetry_coeff", load_cloud_table("kFu96Asymmetry.txt", {25, 4}));
  fu96_delta_coeff = register_buffer(
      "fu96_delta_coeff", load_cloud_table("kFu96Delta.txt", {25, 4}));
  fu98_wavelength = register_buffer(
      "fu98_wavelength", load_cloud_table("kFu98Wavelength.txt", {36}));
  fu98_extinction_coeff =
      register_buffer("fu98_extinction_coeff",
                      load_cloud_table("kFu98Extinction.txt", {36, 3}));
  fu98_absorption_coeff =
      register_buffer("fu98_absorption_coeff",
                      load_cloud_table("kFu98Absorption.txt", {36, 4}));
  fu98_asymmetry_coeff = register_buffer(
      "fu98_asymmetry_coeff", load_cloud_table("kFu98Asymmetry.txt", {36, 4}));
}

torch::Tensor FuWaterIceImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(conc.dim() == 3,
              "Fu water ice expects conc shape (ncol, nlyr, nspecies); got ",
              conc.sizes());
  auto const species_id = options->species_ids()[0];
  TORCH_CHECK(species_id < conc.size(2),
              "Invalid Fu water-ice species_id: ", species_id,
              " for conc with ", conc.size(2), " species");
  TORCH_CHECK(species_id < species_weights.size(),
              "Fu water-ice species_id has no molecular weight: ", species_id);
  auto const ice_conc = conc.select(-1, species_id);
  TORCH_CHECK(torch::all(torch::isfinite(ice_conc)).item<bool>() &&
                  torch::all(ice_conc >= 0.0).item<bool>(),
              "Fu water-ice concentration must be finite and nonnegative");

  torch::Tensor wavelength;
  if (kwargs.count("wavelength") > 0) {
    wavelength = kwargs.at("wavelength");
  } else if (kwargs.count("wavenumber") > 0) {
    auto wavenumber = kwargs.at("wavenumber");
    TORCH_CHECK(torch::all(torch::isfinite(wavenumber)).item<bool>() &&
                    torch::all(wavenumber > 0.0).item<bool>(),
                "Fu water-ice wavenumber must be finite and positive");
    wavelength = 1.0e4 / wavenumber;
  } else {
    TORCH_CHECK(false,
                "Fu water ice requires wavenumber [cm^-1] or wavelength [um]");
  }
  TORCH_CHECK(wavelength.dim() == 1,
              "Fu water ice expects a 1D spectral grid; got ",
              wavelength.sizes());
  TORCH_CHECK(torch::all(torch::isfinite(wavelength)).item<bool>() &&
                  torch::all(wavelength > 0.0).item<bool>(),
              "Fu water-ice wavelength must be finite and positive");
  wavelength = wavelength.to(conc.options());
  auto const use_fu96 = (wavelength >= 0.25) & (wavelength < 4.0);
  auto const use_fu98 = (wavelength >= 4.0) & (wavelength <= 100.0);
  auto const supported = use_fu96 | use_fu98;
  // Evaluate unsupported points at a safe table endpoint, then mask every
  // returned optical property to zero below 0.25 um and above 100 um.
  auto const lookup_wavelength = wavelength.clamp(0.25, 100.0);
  auto const lookup_fu96 = lookup_wavelength < 4.0;

  auto const iwc =
      ice_conc * (1000.0 * species_weights.at(species_id));  // g/m^3
  torch::Tensor dge;
  if (kwargs.count("re") > 0) {
    auto const re = layer_field(kwargs.at("re"), conc, "re");
    TORCH_CHECK(torch::all(torch::isfinite(re)).item<bool>() &&
                    torch::all(re > 0.0).item<bool>(),
                "Fu water-ice re must be finite and positive");
    dge = kReToDge * re;
  } else {
    TORCH_CHECK(kwargs.count("temp") > 0,
                "Fu water ice requires temp [K] when re is not provided");
    auto const temp = layer_field(kwargs.at("temp"), conc, "temp");
    TORCH_CHECK(torch::all(torch::isfinite(temp)).item<bool>() &&
                    torch::all(temp > 0.0).item<bool>(),
                "Fu water-ice temp must be finite and positive");

    // Particle size is undefined in ice-free layers. Give those layers a
    // harmless in-range value so the polynomial evaluation stays finite;
    // their extinction is zero because IWC is zero.
    auto const active = iwc > 0.0;
    auto const safe_iwc = torch::where(active, iwc, torch::ones_like(iwc));
    auto const diagnosed_dge = sun_rikus_dge(temp, safe_iwc);
    dge = torch::where(active, diagnosed_dge,
                       torch::full_like(diagnosed_dge, kSafeDge));
    TORCH_CHECK(torch::all(torch::isfinite(dge)).item<bool>() &&
                    torch::all(dge > 0.0).item<bool>(),
                "Sun-Rikus/Sun (2001) diagnosed a non-finite or nonpositive "
                "Fu generalized effective size; provide re explicitly or "
                "check temp and IWC");
  }

  if (torch::any(iwc > 0.0).item<bool>()) {
    auto const active = (iwc > 0.0).unsqueeze(0);
    auto const d = dge.unsqueeze(0);
    auto const fu96 = use_fu96.view({-1, 1, 1});
    auto const invalid96 =
        active & fu96 & ((d < kFu96MinDge) | (d > kFu96MaxDge));
    auto const fu98 = use_fu98.view({-1, 1, 1});
    auto const invalid98 =
        active & fu98 & ((d < kFu98MinDge) | (d > kFu98MaxDge));
    TORCH_CHECK(!torch::any(invalid96).item<bool>(),
                "Fu96 Dge must be in [18.63, 130.24] um (equivalent re "
                "[12.1005, 84.5934] um) where IWC > 0");
    TORCH_CHECK(!torch::any(invalid98).item<bool>(),
                "Fu98 Dge must be in [11.0, 129.6] um (equivalent re "
                "[7.14471, 84.1777] um) where IWC > 0");
  }

  auto const d = dge.unsqueeze(0);

  // Fu96: piecewise-constant wavelength intervals.
  auto rows96 = torch::zeros(
      {wavelength.size(0)},
      torch::TensorOptions().dtype(torch::kLong).device(wavelength.device()));
  auto const edges96 = fu96_wavelength_edges.to(conc.options());
  for (int band = 0; band < 25; ++band) {
    auto const mask = (lookup_wavelength >= edges96[band]) &
                      (lookup_wavelength < edges96[band + 1]);
    rows96.masked_fill_(mask, band);
  }
  auto const a96 =
      fu96_extinction_coeff.to(conc.options()).index_select(0, rows96);
  auto const b96 =
      fu96_coalbedo_coeff.to(conc.options()).index_select(0, rows96);
  auto const c96 =
      fu96_asymmetry_coeff.to(conc.options()).index_select(0, rows96);
  auto const q96 = fu96_delta_coeff.to(conc.options()).index_select(0, rows96);
  auto extinction96 = fu96_extinction(a96, d);
  auto ssa96 = 1.0 - fu96_coalbedo(b96, d);
  auto g96 = fu96_asymmetry(c96, d);

  bool delta_scale = false;
  if (kwargs.count("fu_delta_scale") > 0) {
    auto flag = kwargs.at("fu_delta_scale");
    TORCH_CHECK(flag.numel() == 1,
                "fu_delta_scale must be a scalar boolean tensor");
    delta_scale = flag.item<bool>();
  }
  if (delta_scale) {
    // Fu (1996), Eq. (3.8): f = 1 / (2 omega) + f_delta. The subsequent
    // scaling follows Appendix A, including computing fw before limiting f
    // by g.
    auto f96 = 0.5 / ssa96 + fu96_delta_fraction(q96, d);
    auto const fw96 = f96 * ssa96;
    f96 = torch::minimum(f96, g96);
    extinction96 = (1.0 - fw96) * extinction96;
    ssa96 = (1.0 - f96) * ssa96 / (1.0 - fw96);
    g96 = (g96 - f96) / (1.0 - f96);
  }

  // Fu98: evaluate at surrounding wavelength nodes and interpolate the
  // optical properties linearly.
  auto lower98 = torch::zeros_like(rows96);
  auto upper98 = torch::ones_like(rows96);
  auto const nodes98 = fu98_wavelength.to(conc.options());
  for (int node = 0; node < 35; ++node) {
    auto const mask = (lookup_wavelength >= nodes98[node]) &
                      (node == 34 ? lookup_wavelength <= nodes98[node + 1]
                                  : lookup_wavelength < nodes98[node + 1]);
    lower98.masked_fill_(mask, node);
    upper98.masked_fill_(mask, node + 1);
  }
  auto const wave_lower = nodes98.index_select(0, lower98);
  auto const wave_upper = nodes98.index_select(0, upper98);
  auto const interp =
      ((lookup_wavelength - wave_lower) / (wave_upper - wave_lower))
          .view({-1, 1, 1});

  auto const a98 = fu98_extinction_coeff.to(conc.options());
  auto const b98 = fu98_absorption_coeff.to(conc.options());
  auto const c98 = fu98_asymmetry_coeff.to(conc.options());
  auto const ext_lower = fu98_extinction(a98.index_select(0, lower98), d);
  auto const ext_upper = fu98_extinction(a98.index_select(0, upper98), d);
  auto const extinction98 = ext_lower + interp * (ext_upper - ext_lower);
  auto const abs_lower = fu98_absorption(b98.index_select(0, lower98), d);
  auto const abs_upper = fu98_absorption(b98.index_select(0, upper98), d);
  auto const absorption98 = abs_lower + interp * (abs_upper - abs_lower);
  auto const g_lower = fu98_asymmetry(c98.index_select(0, lower98), d);
  auto const g_upper = fu98_asymmetry(c98.index_select(0, upper98), d);
  auto const g98 = g_lower + interp * (g_upper - g_lower);
  auto const ssa98 = (extinction98 - absorption98) / extinction98;

  auto const fu96_mask = lookup_fu96.view({-1, 1, 1});
  auto const support_mask = supported.view({-1, 1, 1});
  auto const extinction = torch::where(
      support_mask, torch::where(fu96_mask, extinction96, extinction98), 0.0);
  // The published Fu98 polynomial fit slightly overshoots absorption >
  // extinction at Dge=11 um near 65--75 um. Keep solver inputs physical while
  // retaining the original coefficients and interpolation above.
  auto const ssa = torch::where(
      support_mask, torch::where(fu96_mask, ssa96, ssa98).clamp(0.0, 1.0), 0.0);
  auto const g = torch::where(
      support_mask,
      torch::where(fu96_mask, g96, g98).clamp(-0.999999, 0.999999), 0.0);

  TORCH_CHECK(torch::all(torch::isfinite(extinction)).item<bool>() &&
                  torch::all(extinction >= 0.0).item<bool>(),
              "Fu water-ice extinction is non-finite or negative");
  TORCH_CHECK(torch::all((ssa >= 0.0) & (ssa <= 1.0)).item<bool>(),
              "Fu water-ice single-scattering albedo is outside [0, 1]");
  TORCH_CHECK(torch::all((g > -1.0) & (g < 1.0)).item<bool>(),
              "Fu water-ice asymmetry parameter is outside (-1, 1)");

  auto result = torch::zeros(
      {wavelength.size(0), conc.size(0), conc.size(1), 2 + options->nmom()},
      conc.options());
  result.select(-1, disort::IEX).copy_(extinction * iwc.unsqueeze(0));
  result.select(-1, disort::ISS).copy_(ssa);
  result.narrow(-1, disort::IPM, options->nmom())
      .copy_(henyey_greenstein(options->nmom(), g));
  return result;
}

}  // namespace harp
