#pragma once

// C/C++
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

// base
#include <configure.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/constants.h>

#include <harp/utils/find_resource.hpp>

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {

inline std::string trim_copy(std::string value) {
  auto const begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  auto const end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

inline std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

inline std::string normalize_token(std::string value) {
  value = lower_copy(value);

  std::string out;
  out.reserve(value.size());
  bool last_was_sep = false;
  for (unsigned char c : value) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(c));
      last_was_sep = false;
    } else if (!last_was_sep) {
      out.push_back('_');
      last_was_sep = true;
    }
  }

  while (!out.empty() && out.front() == '_') out.erase(out.begin());
  while (!out.empty() && out.back() == '_') out.pop_back();
  return out;
}

#ifdef NETCDFOUTPUT
inline void check_nc(int status, std::string const& context) {
  TORCH_CHECK(status == NC_NOERR, context, ": ", nc_strerror(status));
}

inline int open_file(std::string const& filename) {
  int fileid = -1;
  auto full_path = find_resource(filename);
  check_nc(nc_open(full_path.c_str(), NC_NOWRITE, &fileid),
           "Failed to open NetCDF file " + full_path);
  return fileid;
}

inline bool try_find_varid(int fileid, std::string const& varname, int* varid) {
  auto const err = nc_inq_varid(fileid, varname.c_str(), varid);
  if (err == NC_NOERR) return true;
  TORCH_CHECK(err == NC_ENOTVAR, "Failed to query variable ", varname, ": ",
              nc_strerror(err));
  return false;
}

inline std::string read_var_units(int fileid, int varid) {
  size_t len = 0;
  auto const status = nc_inq_attlen(fileid, varid, "units", &len);
  if (status == NC_ENOTATT) return "";
  check_nc(status, "Failed to read units attribute length");

  std::string units(len, '\0');
  check_nc(nc_get_att_text(fileid, varid, "units", units.data()),
           "Failed to read units attribute");
  return trim_copy(units);
}

inline std::vector<std::string> read_var_dim_names(int fileid, int varid) {
  int ndims = 0;
  check_nc(nc_inq_varndims(fileid, varid, &ndims),
           "Failed to query variable rank");

  std::vector<int> dimids(ndims);
  check_nc(nc_inq_vardimid(fileid, varid, dimids.data()),
           "Failed to query variable dimensions");

  std::vector<std::string> names(ndims);
  for (int i = 0; i < ndims; ++i) {
    char dim_name[NC_MAX_NAME + 1] = {};
    check_nc(nc_inq_dimname(fileid, dimids[i], dim_name),
             "Failed to query dimension name");
    names[i] = dim_name;
  }
  return names;
}

inline torch::Tensor read_1d_variable(int fileid, std::string const& varname) {
  int varid = -1;
  check_nc(nc_inq_varid(fileid, varname.c_str(), &varid),
           "Missing required variable " + varname);

  int ndims = 0;
  check_nc(nc_inq_varndims(fileid, varid, &ndims),
           "Failed to query variable rank");
  TORCH_CHECK(ndims == 1, "Variable ", varname, " must be 1D, got rank ",
              ndims);

  size_t len = 0;
  int dimid = -1;
  check_nc(nc_inq_vardimid(fileid, varid, &dimid),
           "Failed to query variable dimensions");
  check_nc(nc_inq_dimlen(fileid, dimid, &len),
           "Failed to query variable length");

  std::vector<double> data(len);
  check_nc(nc_get_var_double(fileid, varid, data.data()),
           "Failed to read variable " + varname);
  return torch::tensor(data, torch::kFloat64);
}

inline torch::Tensor read_tensor_permuted(
    int fileid, std::string const& varname,
    std::vector<std::string> const& target_dims) {
  int varid = -1;
  check_nc(nc_inq_varid(fileid, varname.c_str(), &varid),
           "Missing required variable " + varname);

  auto const source_dims = read_var_dim_names(fileid, varid);
  TORCH_CHECK(source_dims.size() == target_dims.size(), "Variable ", varname,
              " has rank ", source_dims.size(), ", expected ",
              target_dims.size());

  std::vector<int> dimids(source_dims.size());
  check_nc(nc_inq_vardimid(fileid, varid, dimids.data()),
           "Failed to query variable dimensions");

  std::vector<int64_t> shape(source_dims.size(), 0);
  for (size_t i = 0; i < dimids.size(); ++i) {
    size_t len = 0;
    check_nc(nc_inq_dimlen(fileid, dimids[i], &len),
             "Failed to query dimension length");
    shape[i] = static_cast<int64_t>(len);
  }

  auto numel = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                               std::multiplies<int64_t>());
  std::vector<double> data(numel);
  check_nc(nc_get_var_double(fileid, varid, data.data()),
           "Failed to read variable " + varname);

  auto tensor = torch::from_blob(data.data(), shape, torch::kFloat64).clone();

  std::vector<int64_t> permute(target_dims.size(), -1);
  for (size_t target = 0; target < target_dims.size(); ++target) {
    auto it =
        std::find(source_dims.begin(), source_dims.end(), target_dims[target]);
    TORCH_CHECK(it != source_dims.end(), "Variable ", varname,
                " is missing dimension ", target_dims[target]);
    permute[target] = static_cast<int64_t>(it - source_dims.begin());
  }

  for (size_t i = 0; i < permute.size(); ++i) {
    TORCH_CHECK(std::count(permute.begin(), permute.end(), permute[i]) == 1,
                "Variable ", varname,
                " dimensions cannot be mapped uniquely to target order");
  }

  return tensor.permute(permute).contiguous();
}
#endif

inline torch::Tensor convert_wavenumber_to_cm1(torch::Tensor values,
                                               std::string const& units,
                                               std::string const& varname) {
  auto const u = lower_copy(trim_copy(units));
  if (u.empty() || u == "cm^-1" || u == "cm-1" || u == "1/cm") return values;
  if (u == "m^-1" || u == "m-1" || u == "1/m") return values * 1.0e-2;
  TORCH_CHECK(false, "Unsupported wavenumber units for ", varname, ": ", units);
  return values;
}

inline torch::Tensor convert_pressure_to_pa(torch::Tensor values,
                                            std::string const& units,
                                            std::string const& varname) {
  auto const u = lower_copy(trim_copy(units));
  if (u.empty() || u == "pa") return values;
  if (u == "bar") return values * 1.0e5;
  if (u == "atm") return values * 101325.0;
  TORCH_CHECK(false, "Unsupported pressure units for ", varname, ": ", units);
  return values;
}

inline torch::Tensor convert_temperature_to_k(torch::Tensor values,
                                              std::string const& units,
                                              std::string const& varname) {
  auto const u = lower_copy(trim_copy(units));
  if (u.empty() || u == "k" || u == "kelvin") return values;
  if (u == "c" || u == "degc" || u == "celsius") return values + 273.15;
  TORCH_CHECK(false, "Unsupported temperature units for ", varname, ": ",
              units);
  return values;
}

inline torch::Tensor apply_positive_fill(torch::Tensor values,
                                         std::string const& quantity_name) {
  auto const positive_mask = values > 0;
  double fill_value = 1.0e-300;
  if (positive_mask.any().item<bool>()) {
    fill_value = values.masked_select(positive_mask).min().item<double>();
  }

  TORCH_CHECK(std::isfinite(fill_value) && fill_value > 0.0,
              "Invalid positive fill value for ", quantity_name, ": ",
              fill_value);

  return torch::where(positive_mask, values,
                      torch::full_like(values, fill_value));
}

inline torch::Tensor convert_line_cross_section_to_m2_per_mol(
    torch::Tensor values, std::string const& units,
    std::string const& varname) {
  auto const u = lower_copy(trim_copy(units));
  if (u == "cm^2 molecule^-1" || u == "cm^2/molecule") {
    return values * (1.0e-4 * constants::Avogadro);
  }
  if (u == "m^2 mol^-1" || u == "m^2/mol") return values;
  if (u == "m^2 kmol^-1" || u == "m^2/kmol") return values * 1.0e-3;
  TORCH_CHECK(false, "Unsupported line/continuum cross-section units for ",
              varname, ": ", units);
  return values;
}

inline torch::Tensor convert_binary_cross_section_to_m5_per_mol2(
    torch::Tensor values, std::string const& units,
    std::string const& varname) {
  auto const u = lower_copy(trim_copy(units));
  if (u == "cm^5 molecule^-2" || u == "cm^5/molecule^2") {
    return values * (1.0e-10 * constants::Avogadro * constants::Avogadro);
  }
  if (u == "m^5 mol^-2" || u == "m^5/mol^2") return values;
  TORCH_CHECK(false, "Unsupported CIA binary coefficient units for ", varname,
              ": ", units);
  return values;
}

}  // namespace harp
