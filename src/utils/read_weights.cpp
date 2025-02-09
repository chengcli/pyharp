// base
#include <configure.h>

// harp
#include <utils/find_resource.hpp>

#include "read_weights.hpp"

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {

torch::Tensor read_weights_rfm(std::string const& filename) {
  auto full_path = find_resource(filename);

#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  size_t len;
  nc_open(full_path.c_str(), NC_NETCDF4, &fileid);

  err = nc_inq_dimid(fileid, "weights", &dimid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_dimlen(fileid, dimid, &len);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_varid(fileid, "weights", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  torch::Tensor out = torch::empty({(int)len}, torch::kFloat64);

  err = nc_get_var_double(fileid, varid, out.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  nc_close(fileid);

  return out;
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif
}

}  // namespace harp
