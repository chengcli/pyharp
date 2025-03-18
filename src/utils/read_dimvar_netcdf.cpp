// C/C++
#include <stdexcept>

// base
#include <configure.h>

// harp
#include <harp/utils/find_resource.hpp>

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {

std::vector<double> read_dimvar_netcdf_double(std::string const& filename,
                                              std::string const& varname) {
  auto full_path = find_resource(filename);

#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  size_t len;
  nc_open(full_path.c_str(), NC_NETCDF4, &fileid);

  err = nc_inq_dimid(fileid, varname.c_str(), &dimid);
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  err = nc_inq_dimlen(fileid, dimid, &len);
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  err = nc_inq_varid(fileid, varname.c_str(), &varid);
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  std::vector<double> out(len);
  err = nc_get_var_double(fileid, varid, out.data());
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  nc_close(fileid);

  return out;
#else
  throw std::runtime_error("NetCDF support is not enabled");
#endif
}

std::vector<float> read_dimvar_netcdf_float(std::string const& filename,
                                            std::string const& varname) {
  auto full_path = find_resource(filename);

#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  size_t len;
  nc_open(full_path.c_str(), NC_NETCDF4, &fileid);

  err = nc_inq_dimid(fileid, varname.c_str(), &dimid);
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  err = nc_inq_dimlen(fileid, dimid, &len);
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  err = nc_inq_varid(fileid, varname.c_str(), &varid);
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  std::vector<float> out(len);
  err = nc_get_var_float(fileid, varid, out.data());
  if (err != NC_NOERR) throw std::runtime_error(nc_strerror(err));

  nc_close(fileid);

  return out;
#else
  throw std::runtime_error("NetCDF support is not enabled");
#endif
}

}  // namespace harp
