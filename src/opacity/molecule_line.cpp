// harp
#include "molecule_line.hpp"

#include <harp/constants.h>

#include <harp/math/interpolation.hpp>
#include <harp/utils/netcdf_opacity_utils.hpp>

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
  int fileid = open_file(options->opacity_files()[0]);

  int wavenumber_id = -1;
  check_nc(nc_inq_varid(fileid, "wavenumber", &wavenumber_id),
           "Missing required variable wavenumber");
  wavenumber = convert_wavenumber_to_cm1(read_1d_variable(fileid, "wavenumber"),
                                         read_var_units(fileid, wavenumber_id),
                                         "wavenumber");

  int pressure_id = -1;
  check_nc(nc_inq_varid(fileid, "pressure", &pressure_id),
           "Missing required variable pressure");
  ln_pressure =
      convert_pressure_to_pa(read_1d_variable(fileid, "pressure"),
                             read_var_units(fileid, pressure_id), "pressure")
          .log();

  int del_temp_id = -1;
  check_nc(nc_inq_varid(fileid, "del_temperature", &del_temp_id),
           "Missing required variable del_temperature");
  temperature_anomaly = convert_temperature_to_k(
      read_1d_variable(fileid, "del_temperature"),
      read_var_units(fileid, del_temp_id), "del_temperature");

  int base_temp_id = -1;
  check_nc(nc_inq_varid(fileid, "temperature", &base_temp_id),
           "Missing required variable temperature");
  ln_temperature_base = convert_temperature_to_k(
                            read_1d_variable(fileid, "temperature"),
                            read_var_units(fileid, base_temp_id), "temperature")
                            .log()
                            .unsqueeze(-1);

  auto const species_token =
      normalize_token(species_names.at(options->species_ids().at(0)));
  auto const line_name = "sigma_line_" + species_token;

  int line_varid = -1;
  check_nc(nc_inq_varid(fileid, line_name.c_str(), &line_varid),
           "Missing required variable " + line_name);
  auto sigma_cross =
      convert_line_cross_section_to_m2_per_mol(
          read_tensor_permuted(fileid, line_name,
                               {"wavenumber", "pressure", "del_temperature"}),
          read_var_units(fileid, line_varid), line_name)
          .unsqueeze(-1);

  int nvars = 0;
  check_nc(nc_inq_nvars(fileid, &nvars), "Failed to query variable count");
  auto const continuum_prefix = "sigma_continuum_" + species_token + "_";
  auto const self_continuum_name = continuum_prefix + "self_mt_ckd";
  auto const foreign_continuum_name = continuum_prefix + "foreign_mt_ckd";
  auto const legacy_continuum_name = continuum_prefix + "mt_ckd";
  torch::Tensor sigma_continuum_self;
  torch::Tensor sigma_continuum_foreign;
  torch::Tensor sigma_continuum_legacy;
  for (int i = 0; i < nvars; ++i) {
    char name[NC_MAX_NAME + 1] = {};
    check_nc(nc_inq_varname(fileid, i, name), "Failed to query variable name");
    std::string varname(name);
    if (varname.rfind(continuum_prefix, 0) != 0) continue;

    auto continuum =
        convert_line_cross_section_to_m2_per_mol(
            read_tensor_permuted(fileid, varname,
                                 {"wavenumber", "pressure", "del_temperature"}),
            read_var_units(fileid, i), varname)
            .unsqueeze(-1);
    if (varname == self_continuum_name) {
      sigma_continuum_self = continuum;
    } else if (varname == foreign_continuum_name) {
      sigma_continuum_foreign = continuum;
    } else if (varname == legacy_continuum_name) {
      sigma_continuum_legacy = continuum;
    } else {
      sigma_cross += continuum;
    }
  }

  auto const has_self = sigma_continuum_self.defined();
  auto const has_foreign = sigma_continuum_foreign.defined();
  TORCH_CHECK(has_self == has_foreign, "Split H2O continuum requires both ",
              self_continuum_name, " and ", foreign_continuum_name);
  has_split_h2o_continuum = has_self && has_foreign;
  if (has_split_h2o_continuum) {
    ln_sigma_continuum_self =
        apply_positive_fill(sigma_continuum_self, self_continuum_name).log();
    ln_sigma_continuum_foreign =
        apply_positive_fill(sigma_continuum_foreign, foreign_continuum_name)
            .log();
  } else if (sigma_continuum_legacy.defined()) {
    sigma_cross += sigma_continuum_legacy;
  }

  sigma_cross = apply_positive_fill(sigma_cross, line_name);
  ln_sigma_cross = sigma_cross.log();

  check_nc(nc_close(fileid), "Failed to close NetCDF file");
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif

  // register all buffers
  register_buffer("wavenumber", wavenumber);
  register_buffer("ln_pressure", ln_pressure);
  register_buffer("temperature_anomaly", temperature_anomaly);
  register_buffer("ln_sigma_cross", ln_sigma_cross);
  register_buffer("ln_temperature_base", ln_temperature_base);
  if (has_split_h2o_continuum) {
    register_buffer("ln_sigma_continuum_self", ln_sigma_continuum_self);
    register_buffer("ln_sigma_continuum_foreign", ln_sigma_continuum_foreign);
  }
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
  auto temperature_base =
      interpn({lnp}, {ln_pressure}, ln_temperature_base).squeeze(-1).exp();
  auto tempa = temp - temperature_base;

  int const nwave = wave_query.size(0);
  auto wave =
      wave_query.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  lnp = lnp.unsqueeze(0).expand({nwave, ncol, nlyr});
  tempa = tempa.unsqueeze(0).expand({nwave, ncol, nlyr});

  // Clamp queries to the tabulated bounds. Extrapolating logarithmic line
  // cross sections can produce nonphysical opacity outside the table coverage.
  auto query = std::vector<torch::Tensor>{wave, lnp, tempa};
  auto coordinates =
      std::vector<torch::Tensor>{wavenumber, ln_pressure, temperature_anomaly};
  auto out = interpn(query, coordinates, ln_sigma_cross, false).exp();

  // Check species id in range
  TORCH_CHECK(options->species_ids()[0] >= 0 &&
                  options->species_ids()[0] < conc.size(2),
              "Invalid species_id: ", options->species_ids()[0]);

  auto water_conc = conc.select(-1, options->species_ids()[0]);
  if (has_split_h2o_continuum) {
    auto total_conc = conc.sum(-1);
    TORCH_CHECK(torch::all(total_conc > 0.0).item<bool>(),
                "Total gas concentration must be positive");
    TORCH_CHECK(torch::all(water_conc >= 0.0).item<bool>(),
                "H2O concentration must be non-negative");
    auto h2o_vmr = water_conc / total_conc;
    auto sigma_self =
        interpn(query, coordinates, ln_sigma_continuum_self, false).exp();
    auto sigma_foreign =
        interpn(query, coordinates, ln_sigma_continuum_foreign, false).exp();
    out += h2o_vmr.unsqueeze(0).unsqueeze(-1) * sigma_self;
    out += (1.0 - h2o_vmr).unsqueeze(0).unsqueeze(-1) * sigma_foreign;
  }

  return out * water_conc.unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
