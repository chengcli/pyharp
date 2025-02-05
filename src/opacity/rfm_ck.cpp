// harp
#include "rfm_ck.hpp"

#include <math/interpolation.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {

RFMCKImpl::RFMCKImpl(AttenuatorOptionis const& options_) : options(options_) {
  TORCH_CHECK(options.opacity_file().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] > 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(options.type() == "rfm_ck", "Invalid type: ", options.type());

  reset();
}

void RFMCKImpl::reset() {
#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  len_[0] = 22;  // number of pressure
  len_[1] = 27;  // number of temperature
  // len_[2] = 26; // number of bands
  len_[2] = 8;  // number of guass points

  nc_open(fname.c_str(), NC_NETCDF4, &fileid);

  axis_.resize(len_[0] + len_[1] + len_[2]);

  nc_inq_varid(fileid, "p", &varid);
  nc_get_var_double(fileid, varid, axis_.data());
  nc_inq_varid(fileid, "t", &varid);
  nc_get_var_double(fileid, varid, axis_.data() + len_[0]);
  nc_inq_varid(fileid, "samples", &varid);
  nc_get_var_double(fileid, varid, axis_.data() + len_[0] + len_[1]);

  kcoeff_.resize(len_[0] * len_[1] * len_[2]);
  nc_inq_varid(fileid, GetName().c_str(), &varid);
  size_t start[4] = {0, 0, (size_t)bid, 0};
  size_t count[4] = {len_[0], len_[1], 1, len_[2]};
  nc_get_vara_double(fileid, varid, start, count, kcoeff_.data());
  nc_close(fileid);
#endif
}

torch::Tensor RFMCKImpl::forward(torch::Tensor xfrac, torch::Tensor pres,
                                 torch::Tensor temp) {
  int nwve = wave.size(0);
  // first axis is wavenumber, second is pressure, third is temperature anomaly
  Real val, coord[3] = {log(var.q[IPR]), var.q[IDN], g1};
  interpn(&val, coord, kcoeff_.data(), axis_.data(), len_, 3, 1);

  auto dens = pres / (Constants::kBoltz * temp);
  return exp(val) * dens;  // ln(m*2/kmol) -> 1/m
}

}  // namespace harp
