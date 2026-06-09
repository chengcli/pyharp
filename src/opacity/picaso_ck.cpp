#include "picaso_ck.hpp"

#include <harp/math/interpolation.hpp>
#include <harp/utils/netcdf_opacity_utils.hpp>

namespace harp {

PicasoCKImpl::PicasoCKImpl(OpacityOptions const& options_) : options(options_) {
  TORCH_CHECK(options->opacity_files().size() == 1,
              "Only one opacity file is allowed");
  TORCH_CHECK(options->type().empty() || options->type() == "picaso-ck",
              "Mismatch opacity type: ", options->type(),
              " expecting 'picaso-ck'");
  reset();
}

void PicasoCKImpl::reset() {
#ifdef NETCDFOUTPUT
  int fileid = open_file(options->opacity_files()[0]);

  auto qwave =
      read_tensor_permuted(fileid, "quadrature_wavenumber", {"band", "gpoint"});
  auto qweight =
      read_tensor_permuted(fileid, "quadrature_weight", {"band", "gpoint"});
  auto lower = read_1d_variable(fileid, "wavenumber_lower");
  auto upper = read_1d_variable(fileid, "wavenumber_upper");
  auto ng = qwave.size(1);

  wavenumber = qwave.flatten();
  weights = qweight.flatten();
  wave_lower = lower.unsqueeze(1).expand({lower.size(0), ng}).flatten();
  wave_upper = upper.unsqueeze(1).expand({upper.size(0), ng}).flatten();
  ln_pressure = read_1d_variable(fileid, "pressure").log();
  temperature_anomaly = read_1d_variable(fileid, "temperature_offset");
  ln_temperature_base =
      read_1d_variable(fileid, "nominal_temperature").log().unsqueeze(-1);
  ln_sigma_cross =
      read_tensor_permuted(fileid, "kappa",
                           {"band", "gpoint", "pressure", "temperature_offset"})
          .reshape({-1, ln_pressure.size(0), temperature_anomaly.size(0), 1})
          .clamp_min(1.e-300)
          .log();

  check_nc(nc_close(fileid), "Failed to close NetCDF file");
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif

  register_buffer("wavenumber", wavenumber);
  register_buffer("weights", weights);
  register_buffer("wave_lower", wave_lower);
  register_buffer("wave_upper", wave_upper);
  register_buffer("ln_pressure", ln_pressure);
  register_buffer("temperature_anomaly", temperature_anomaly);
  register_buffer("ln_temperature_base", ln_temperature_base);
  register_buffer("ln_sigma_cross", ln_sigma_cross);
}

torch::Tensor PicasoCKImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");
  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");
  auto wave_query =
      kwargs.count("wavenumber") > 0 ? kwargs.at("wavenumber") : wavenumber;

  auto lnp = pres.log();
  auto temperature_base =
      interpn({lnp}, {ln_pressure}, ln_temperature_base).squeeze(-1).exp();
  auto tempa = temp - temperature_base;
  int const nwave = wave_query.size(0);
  auto wave = wave_query.unsqueeze(-1).unsqueeze(-1).expand(
      {nwave, conc.size(0), conc.size(1)});
  lnp = lnp.unsqueeze(0).expand_as(wave);
  tempa = tempa.unsqueeze(0).expand_as(wave);
  auto sigma =
      interpn({wave, lnp, tempa},
              {wavenumber, ln_pressure, temperature_anomaly}, ln_sigma_cross)
          .exp();

  // PICASO pre-mixed molecular opacity [cm^2/mol] times total concentration.
  return 1.e-4 * sigma * conc.sum(-1).unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
