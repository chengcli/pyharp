#pragma once

// C/C++
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace harp {

//! read a dimension variable from a netCDF file
/*!
 * This function reads a dimension variable from a netCDF file and returns
 * it as a std::vector.
 *
 * The netCDF file must contain a variable with the given name, and the
 * variable must have the same name as the dimension (dimension variable).
 *
 * \param filename the name of the netCDF file
 * \param varname the name of the dimension variable
 */
std::vector<double> read_dimvar_netcdf_double(std::string const& filename,
                                              std::string const& varname);

//! read a dimension variable from a netCDF file
/*!
 * This function reads a dimension variable from a netCDF file and returns
 * it as a std::vector.
 *
 * The netCDF file must contain a variable with the given name, and the
 * variable must have the same name as the dimension (dimension variable).
 *
 * \param filename the name of the netCDF file
 * \param varname the name of the dimension variable
 */
std::vector<float> read_dimvar_netcdf_float(std::string const& filename,
                                            std::string const& varname);

//! wrapper function for read_dimvar_netcdf_double and read_dimvar_netcdf_float
template <typename T>
std::vector<T> read_dimvar_netcdf(std::string const& filename,
                                  std::string const& varname) {
  if constexpr (std::is_same_v<T, double>) {
    return read_dimvar_netcdf_double(filename, varname);
  } else if constexpr (std::is_same_v<T, float>) {
    return read_dimvar_netcdf_float(filename, varname);
  } else {
    throw std::runtime_error("Unsupported type for dimension variable");
  }
}

}  // namespace harp
