// harp
#include "molecule_line.hpp"

#include <harp/constants.h>

#include <harp/math/interpolation.hpp>

#include "netcdf_opacity_utils.hpp"

namespace harp {

extern std::vector<std::string> species_names;

MoleculeLineImpl::MoleculeLineImpl(OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->opacity_files().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options->species_ids().size() == 1,
              "Only one species is allowed");

  TORCH_CHECK(options->species_ids()[0] >= 0,
              "Invalid species_id: ", options->species_ids()[0]);

  TORCH_CHECK(options->type().empty() || options->type() == "molecule-line",
              "Mismatch opacity type: ", options->type(),
              " expecting 'molecule-line'");

  reset();
}

void MoleculeLineImpl::reset() {
#ifdef NETCDFOUTPUT
  int fileid = opacity_netcdf::open_file(options->opacity_files()[0]);

  int wavenumber_id = -1;
  opacity_netcdf::check_nc(nc_inq_varid(fileid, "wavenumber", &wavenumber_id),
                           "Missing required variable wavenumber");
  wavenumber = opacity_netcdf::convert_wavenumber_to_cm1(
      opacity_netcdf::read_1d_variable(fileid, "wavenumber"),
      opacity_netcdf::read_var_units(fileid, wavenumber_id), "wavenumber");

  int pressure_id = -1;
  opacity_netcdf::check_nc(nc_inq_varid(fileid, "pressure", &pressure_id),
                           "Missing required variable pressure");
  ln_pressure =
      opacity_netcdf::convert_pressure_to_pa(
          opacity_netcdf::read_1d_variable(fileid, "pressure"),
          opacity_netcdf::read_var_units(fileid, pressure_id), "pressure")
          .log();

  int del_temp_id = -1;
  opacity_netcdf::check_nc(
      nc_inq_varid(fileid, "del_temperature", &del_temp_id),
      "Missing required variable del_temperature");
  temperature_anomaly = opacity_netcdf::convert_temperature_to_k(
      opacity_netcdf::read_1d_variable(fileid, "del_temperature"),
      opacity_netcdf::read_var_units(fileid, del_temp_id), "del_temperature");

  int base_temp_id = -1;
  opacity_netcdf::check_nc(nc_inq_varid(fileid, "temperature", &base_temp_id),
                           "Missing required variable temperature");
  temperature_base =
      opacity_netcdf::convert_temperature_to_k(
          opacity_netcdf::read_1d_variable(fileid, "temperature"),
          opacity_netcdf::read_var_units(fileid, base_temp_id), "temperature")
          .unsqueeze(-1);

  auto const species_token = opacity_netcdf::normalize_token(
      species_names.at(options->species_ids().at(0)));
  auto const line_name = "sigma_line_" + species_token;

  int line_varid = -1;
  opacity_netcdf::check_nc(nc_inq_varid(fileid, line_name.c_str(), &line_varid),
                           "Missing required variable " + line_name);
  ln_sigma_cross =
      opacity_netcdf::convert_line_cross_section_to_m2_per_mol(
          opacity_netcdf::read_tensor_permuted(
              fileid, line_name, {"wavenumber", "pressure", "del_temperature"}),
          opacity_netcdf::read_var_units(fileid, line_varid), line_name)
          .unsqueeze(-1);

  int nvars = 0;
  opacity_netcdf::check_nc(nc_inq_nvars(fileid, &nvars),
                           "Failed to query variable count");
  auto const continuum_prefix = "sigma_continuum_" + species_token + "_";
  for (int i = 0; i < nvars; ++i) {
    char name[NC_MAX_NAME + 1] = {};
    opacity_netcdf::check_nc(nc_inq_varname(fileid, i, name),
                             "Failed to query variable name");
    std::string varname(name);
    if (varname.rfind(continuum_prefix, 0) != 0) continue;

    ln_sigma_cross +=
        opacity_netcdf::convert_line_cross_section_to_m2_per_mol(
            opacity_netcdf::read_tensor_permuted(
                fileid, varname, {"wavenumber", "pressure", "del_temperature"}),
            opacity_netcdf::read_var_units(fileid, i), varname)
            .unsqueeze(-1);
  }

  opacity_netcdf::check_nc(nc_close(fileid), "Failed to close NetCDF file");
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif

  // register all buffers
  register_buffer("wavenumber", wavenumber);
  register_buffer("ln_pressure", ln_pressure);
  register_buffer("temperature_anomaly", temperature_anomaly);
  register_buffer("ln_sigma_cross", ln_sigma_cross);
  register_buffer("temperature_base", temperature_base);
}

torch::Tensor MoleculeLineImpl::forward(
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

  torch::Tensor wave_query;
  if (kwargs.count("wavenumber") > 0) {
    wave_query = kwargs.at("wavenumber");
  } else if (kwargs.count("wavelength") > 0) {
    wave_query = 1.0e4 / kwargs.at("wavelength");
  } else {
    wave_query = wavenumber;
  }
  TORCH_CHECK(wave_query.dim() == 1,
              "MoleculeLine expects a 1D wavenumber or wavelength grid");

  auto lnp = pres.log();
  auto tempa =
      temp - interpn({lnp}, {ln_pressure}, temperature_base).squeeze(-1);

  int const nwave = wave_query.size(0);
  auto wave =
      wave_query.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  lnp = lnp.unsqueeze(0).expand({nwave, ncol, nlyr});
  tempa = tempa.unsqueeze(0).expand({nwave, ncol, nlyr});

  auto out = interpn({wave, lnp, tempa},
                     {wavenumber, ln_pressure, temperature_anomaly},
                     ln_sigma_cross, true);

  // Check species id in range
  TORCH_CHECK(options->species_ids()[0] >= 0 &&
                  options->species_ids()[0] < conc.size(2),
              "Invalid species_id: ", options->species_ids()[0]);

  return out *
         conc.select(-1, options->species_ids()[0]).unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
