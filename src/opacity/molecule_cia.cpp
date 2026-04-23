// harp
#include "molecule_cia.hpp"

#include <harp/math/interpolation.hpp>
#include <harp/utils/netcdf_opacity_utils.hpp>

namespace harp {

extern std::vector<std::string> species_names;

MoleculeCIAImpl::MoleculeCIAImpl(OpacityOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options->opacity_files().size() == 1,
              "Only one opacity file is allowed");
  TORCH_CHECK(
      options->species_ids().size() == 1 || options->species_ids().size() == 2,
      "MoleculeCIA expects one species for self-CIA or two species for binary "
      "CIA");
  TORCH_CHECK(options->type().empty() || options->type() == "molecule-cia",
              "Mismatch opacity type: ", options->type(),
              " expecting 'molecule-cia'");
  reset();
}

void MoleculeCIAImpl::reset() {
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

  auto first = normalize_token(species_names.at(options->species_ids().at(0)));
  auto second = first;
  if (options->species_ids().size() == 2) {
    second = normalize_token(species_names.at(options->species_ids().at(1)));
  }

  std::vector<std::string> candidates = {"binary_absorption_coefficient_" +
                                         first + "_" + second};
  if (first != second) {
    candidates.push_back("binary_absorption_coefficient_" + second + "_" +
                         first);
  }

  int varid = -1;
  std::string selected;
  for (auto const& candidate : candidates) {
    if (try_find_varid(fileid, candidate, &varid)) {
      selected = candidate;
      break;
    }
  }
  TORCH_CHECK(!selected.empty(), "No CIA variable found for species pair ",
              first, "-", second);

  auto sigma_binary =
      convert_binary_cross_section_to_m5_per_mol2(
          read_tensor_permuted(fileid, selected,
                               {"wavenumber", "pressure", "del_temperature"}),
          read_var_units(fileid, varid), selected)
          .unsqueeze(-1);
  sigma_binary = apply_positive_fill(sigma_binary, selected);
  ln_sigma_binary = sigma_binary.log();

  check_nc(nc_close(fileid), "Failed to close NetCDF file");
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif

  register_buffer("wavenumber", wavenumber);
  register_buffer("ln_pressure", ln_pressure);
  register_buffer("temperature_anomaly", temperature_anomaly);
  register_buffer("ln_sigma_binary", ln_sigma_binary);
  register_buffer("ln_temperature_base", ln_temperature_base);
}

torch::Tensor MoleculeCIAImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  int const ncol = conc.size(0);
  int const nlyr = conc.size(1);
  TORCH_CHECK(pres.size(0) == ncol && pres.size(1) == nlyr,
              "Invalid pres shape: ", pres.sizes());
  TORCH_CHECK(temp.size(0) == ncol && temp.size(1) == nlyr,
              "Invalid temp shape: ", temp.sizes());

  torch::Tensor wave_query;
  if (kwargs.count("wavenumber") > 0) {
    wave_query = kwargs.at("wavenumber");
  } else if (kwargs.count("wavelength") > 0) {
    wave_query = 1.0e4 / kwargs.at("wavelength");
  } else {
    wave_query = wavenumber;
  }
  TORCH_CHECK(wave_query.dim() == 1,
              "MoleculeCIA expects a 1D wavenumber or wavelength grid");

  auto lnp = pres.log();
  auto temperature_base =
      interpn({lnp}, {ln_pressure}, ln_temperature_base).squeeze(-1).exp();
  auto del_temp = temp - temperature_base;

  int const nwave = wave_query.size(0);
  auto wave =
      wave_query.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  auto lnp_grid = lnp.unsqueeze(0).expand({nwave, ncol, nlyr});
  auto temp_grid = del_temp.unsqueeze(0).expand({nwave, ncol, nlyr});

  auto coeff = interpn({wave, lnp_grid, temp_grid},
                       {wavenumber, ln_pressure, temperature_anomaly},
                       ln_sigma_binary, true)
                   .exp();

  auto species0 = conc.select(-1, options->species_ids().at(0));
  auto species1 = conc.select(-1, options->species_ids().size() == 2
                                      ? options->species_ids().at(1)
                                      : options->species_ids().at(0));
  return coeff * species0.unsqueeze(0).unsqueeze(-1) *
         species1.unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
