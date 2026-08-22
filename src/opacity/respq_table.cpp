#include "respq_table.hpp"

#include <harp/math/interpolation.hpp>
#include <harp/utils/netcdf_opacity_utils.hpp>

namespace harp {

namespace {
#ifdef NETCDFOUTPUT
torch::Tensor read_components(int fileid, std::string const& name,
                              std::string const& component_dim,
                              std::string const& expected_units) {
  int varid = -1;
  check_nc(nc_inq_varid(fileid, name.c_str(), &varid),
           "Missing required variable " + name);
  auto const units = normalize_token(read_var_units(fileid, varid));
  TORCH_CHECK(units == normalize_token(expected_units), name,
              " units must be '", expected_units, "'");
  return read_tensor_permuted(
             fileid, name,
             {"pressure", "temperature_offset", "point", component_dim})
      .clamp_min(1.e-300)
      .log();
}
#endif
}  // namespace

RespqTableImpl::RespqTableImpl(OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->opacity_files().size() == 1,
              "Only one opacity file is allowed");
  TORCH_CHECK(options->type().empty() || options->type() == "respq-table",
              "Mismatch opacity type: ", options->type(),
              " expecting 'respq-table'");
  reset();
}

void RespqTableImpl::reset() {
#ifdef NETCDFOUTPUT
  int fileid = open_file(options->opacity_files()[0]);

  wavenumber = read_1d_variable(fileid, "wavenumber");
  wave_lower = read_1d_variable(fileid, "wavenumber_lower");
  wave_upper = read_1d_variable(fileid, "wavenumber_upper");
  weights =
      read_1d_variable(fileid, "quadrature_weight") / (wave_upper - wave_lower);

  ln_pressure = read_1d_variable(fileid, "pressure").log();
  temperature_anomaly = read_1d_variable(fileid, "temperature_offset");
  ln_temperature_base =
      read_1d_variable(fileid, "nominal_temperature").log().unsqueeze(-1);
  reference_mole_fraction = read_1d_variable(fileid, "reference_mole_fraction");

  int varid = -1;
  if (try_find_varid(fileid, "kappa_linear", &varid)) {
    ln_linear =
        read_components(fileid, "kappa_linear", "linear_component", "m2 mol-1");
    nlinear = ln_linear.size(-1);
    ln_linear = ln_linear.reshape(
        {ln_pressure.numel(), temperature_anomaly.numel(), -1});
  }
  if (try_find_varid(fileid, "kappa_binary", &varid)) {
    ln_binary =
        read_components(fileid, "kappa_binary", "binary_component", "m5 mol-2");
    nbinary = ln_binary.size(-1);
    ln_binary = ln_binary.reshape(
        {ln_pressure.numel(), temperature_anomaly.numel(), -1});
  }
  if (try_find_varid(fileid, "scattering_coefficient", &varid)) {
    TORCH_CHECK(normalize_token(read_var_units(fileid, varid)) == "m2_mol_1",
                "scattering_coefficient units must be 'm2 mol-1'");
    scattering = read_1d_variable(fileid, "scattering_coefficient");
    phase_moment =
        read_tensor_permuted(fileid, "phase_moment", {"point", "moment"});
  }

  check_nc(nc_close(fileid), "Failed to close NetCDF file");
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif

  int64_t const npoint = wavenumber.numel();
  TORCH_CHECK(npoint > 0, "ReSPQ table is empty");
  TORCH_CHECK(wave_lower.numel() == npoint && wave_upper.numel() == npoint &&
                  weights.numel() == npoint,
              "ReSPQ spectral arrays must have the same length");
  TORCH_CHECK(torch::all(wave_upper > wave_lower).item<bool>(),
              "ReSPQ points must have positive spectral widths");
  TORCH_CHECK(
      torch::all(torch::isfinite(weights) & (weights >= 0.)).item<bool>(),
      "ReSPQ weights must be finite and non-negative");
  if (npoint > 1) {
    TORCH_CHECK(torch::all(wavenumber.slice(0, 1, npoint) >
                           wavenumber.slice(0, 0, npoint - 1))
                    .item<bool>(),
                "ReSPQ wavenumbers must be strictly increasing");
  }
  TORCH_CHECK(ln_pressure.numel() > 1 &&
                  torch::all(ln_pressure.slice(0, 1, ln_pressure.numel()) >
                             ln_pressure.slice(0, 0, ln_pressure.numel() - 1))
                      .item<bool>(),
              "ReSPQ pressure must be strictly increasing");
  TORCH_CHECK(
      temperature_anomaly.numel() > 1 &&
          torch::all(
              temperature_anomaly.slice(0, 1, temperature_anomaly.numel()) >
              temperature_anomaly.slice(0, 0, temperature_anomaly.numel() - 1))
              .item<bool>(),
      "ReSPQ temperature_offset must be strictly increasing");
  TORCH_CHECK(ln_linear.defined() || ln_binary.defined(),
              "ReSPQ table has no absorption coefficients");
  if (ln_linear.defined()) {
    TORCH_CHECK(ln_linear.size(2) == npoint * nlinear,
                "Invalid kappa_linear shape");
  }
  if (ln_binary.defined()) {
    TORCH_CHECK(ln_binary.size(2) == npoint * nbinary,
                "Invalid kappa_binary shape");
  }
  TORCH_CHECK(options->species_ids().size() ==
                  static_cast<size_t>(reference_mole_fraction.numel()),
              "ReSPQ species and reference composition sizes differ");
  TORCH_CHECK(
      torch::all(reference_mole_fraction >= 0.).item<bool>() &&
          torch::isclose(reference_mole_fraction.sum(),
                         torch::tensor(1., torch::kFloat64), 1.e-8, 1.e-8)
              .item<bool>(),
      "reference_mole_fraction must be non-negative and sum to one");
  if (scattering.defined()) {
    TORCH_CHECK(scattering.dim() == 1 && scattering.numel() == npoint &&
                    torch::all(scattering >= 0.).item<bool>(),
                "Invalid scattering_coefficient");
    TORCH_CHECK(phase_moment.dim() == 2 && phase_moment.size(0) == npoint,
                "phase_moment must have dimensions (point, moment)");
  }

  register_buffer("wavenumber", wavenumber);
  register_buffer("weights", weights);
  register_buffer("wave_lower", wave_lower);
  register_buffer("wave_upper", wave_upper);
  register_buffer("ln_pressure", ln_pressure);
  register_buffer("temperature_anomaly", temperature_anomaly);
  register_buffer("ln_temperature_base", ln_temperature_base);
  if (ln_linear.defined()) register_buffer("ln_linear", ln_linear);
  if (ln_binary.defined()) register_buffer("ln_binary", ln_binary);
  if (scattering.defined()) {
    register_buffer("scattering", scattering);
    register_buffer("phase_moment", phase_moment);
  }
  register_buffer("reference_mole_fraction", reference_mole_fraction);
  bounds_mask = register_buffer("bounds_mask", torch::zeros({0}, torch::kBool));
}

int RespqTableImpl::scattering_moments() const {
  return phase_moment.defined() ? phase_moment.size(1) : 0;
}

torch::Tensor RespqTableImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");
  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  TORCH_CHECK(pres.size(0) == ncol && pres.size(1) == nlyr,
              "Invalid pres shape: ", pres.sizes(),
              "; needs to be (ncol, nlyr)");
  TORCH_CHECK(temp.size(0) == ncol && temp.size(1) == nlyr,
              "Invalid temp shape: ", temp.sizes(),
              "; needs to be (ncol, nlyr)");

  if (kwargs.count("wavenumber") > 0) {
    auto const& wave_query = kwargs.at("wavenumber");
    TORCH_CHECK(wave_query.sizes() == wavenumber.sizes() &&
                    torch::allclose(wave_query, wavenumber),
                "RespqTable uses its frozen spectral points");
  }

  auto ids = torch::tensor(
      options->species_ids(),
      torch::TensorOptions().dtype(torch::kLong).device(conc.device()));
  auto selected = conc.index_select(-1, ids);
  auto total = selected.sum(-1);
  TORCH_CHECK(torch::allclose(total, conc.sum(-1), 1.e-8, 1.e-12),
              "ReSPQ species must cover the atmospheric composition");
  TORCH_CHECK(torch::all(total > 0.).item<bool>(),
              "ReSPQ composition must have positive total concentration");
  auto fraction = selected / total.unsqueeze(-1);
  auto reference = reference_mole_fraction.to(conc.options())
                       .view({1, 1, -1})
                       .expand_as(fraction);
  TORCH_CHECK(torch::allclose(fraction, reference, 1.e-6, 1.e-8),
              "Atmospheric composition does not match the ReSPQ table");

  auto lnp = pres.log();
  auto temperature_base =
      interpn({lnp}, {ln_pressure}, ln_temperature_base, false)
          .squeeze(-1)
          .exp();
  auto tempa = temp - temperature_base;
  bounds_mask.set_((lnp < ln_pressure[0]) | (lnp > ln_pressure[-1]) |
                   (tempa < temperature_anomaly[0]) |
                   (tempa > temperature_anomaly[-1]));

  auto total_q = total.unsqueeze(0);
  auto absorption =
      torch::zeros({wavenumber.numel(), ncol, nlyr}, conc.options());
  if (ln_linear.defined()) {
    auto linear = interpn({lnp, tempa}, {ln_pressure, temperature_anomaly},
                          ln_linear, false)
                      .exp()
                      .view({ncol, nlyr, wavenumber.numel(), nlinear})
                      .sum(-1)
                      .permute({2, 0, 1});
    absorption += linear * total_q;
  }
  if (ln_binary.defined()) {
    auto binary = interpn({lnp, tempa}, {ln_pressure, temperature_anomaly},
                          ln_binary, false)
                      .exp()
                      .view({ncol, nlyr, wavenumber.numel(), nbinary})
                      .sum(-1)
                      .permute({2, 0, 1});
    absorption += binary * total_q.square();
  }
  if (!scattering.defined()) return absorption.unsqueeze(-1);

  auto scattering_alpha =
      scattering.to(conc.options()).view({-1, 1, 1}) * total_q;
  auto extinction = absorption + scattering_alpha;
  auto out =
      torch::zeros({wavenumber.numel(), ncol, nlyr, 2 + scattering_moments()},
                   conc.options());
  out.select(-1, 0) = extinction;
  out.select(-1, 1) = (scattering_alpha / extinction).clamp_max(1. - 1.e-12);
  out.narrow(-1, 2, scattering_moments()) =
      phase_moment.to(conc.options())
          .unsqueeze(1)
          .unsqueeze(1)
          .expand({wavenumber.numel(), ncol, nlyr, scattering_moments()});
  return out;
}

}  // namespace harp
