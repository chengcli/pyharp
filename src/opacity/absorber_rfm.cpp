// harp
#include <constants.h>

#include <utils/find_resource.hpp>

#include "attenuator.hpp"

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {
void AbsorberRFMImpl::reset() {
  kdata = register_buffer(
      "kdata",
      torch::zeros({1 + options.npmom(), options.nspec(), options.atm().ncomp(),
                    options.atm().npres(), options.atm().ntemp()},
                   torch::kFloat));
  scale_grid = register_module("scale_grid", AtmToStandardGrid(options.atm()));
  load();
}

void AbsorberRFMImpl::load() {
  if (options.opacity_file().empty()) return;
  std::string full_path = find_resource(options.opacity_file());

#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  nc_open(full_path.c_str(), NC_NETCDF4, &fileid);

  nc_inq_dimid(fileid, "Wavenumber", &dimid);
  nc_inq_dimlen(fileid, dimid, len_);
  nc_inq_dimid(fileid, "Pressure", &dimid);
  nc_inq_dimlen(fileid, dimid, len_ + 1);
  nc_inq_dimid(fileid, "TempGrid", &dimid);
  nc_inq_dimlen(fileid, dimid, len_ + 2);

  axis_.resize(len_[0] + len_[1] + len_[2]);

  nc_inq_varid(fileid, "Wavenumber", &varid);
  nc_get_var_double(fileid, varid, axis_.data());

  err = nc_inq_varid(fileid, "Pressure", &varid);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_get_var_double(fileid, varid, axis_.data() + len_[0]);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_inq_varid(fileid, "TempGrid", &varid);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  err = nc_get_var_double(fileid, varid, axis_.data() + len_[0] + len_[1]);
  if (err != NC_NOERR) {
    throw std::runtime_error(nc_strerror(err));
  }

  float* temp = new float[len_[1]];
  nc_inq_varid(fileid, "Temperature", &varid);
  nc_get_var_double(fileid, varid, temp);

  refatm_.NewAthenaArray(NHYDRO, len_[1]);
  for (int i = 0; i < len_[1]; i++) {
    refatm_(IPR, i) = axis_[len_[0] + i];
    refatm_(IDN, i) = temp[i];
  }

  kdata = torch::zeros((2 + options.npmom(), options.nspec(), options.npres(),
                        options.ntemp(), options.ncomp()),
                       torch::kFloat);

  nc_inq_varid(fileid, options.name().c_str(), &varid);
  nc_get_var_double(fileid, varid, kdata.data());
  nc_close(fileid);
  delete[] temp;
#endif
}

torch::Tensor AbsorberRFMImpl::forward(torch::Tensor var_x) {
  namespace F = torch::nn::functional;
  auto grid = scale_grid->forward(var_x, options.var_id()[0]);

  // interpolate to model grid
  auto op = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kBorder);
  auto kcross = F::grid_sample(kdata, grid, op);

  auto x0 = var_x[options.var_id()[0]];
  auto dens = x0 * var_x[index::IPR] / (constants::Rgas * var_x[index::ITM]);
  return 1.E-3 * kcross.exp() * dens;  // ln(m^2/kmol) -> 1/m
}

}  // namespace harp
