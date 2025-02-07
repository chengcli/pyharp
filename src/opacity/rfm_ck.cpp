// base
#include <configure.h>

// harp
#include <math/interpolation.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

#include "rfm_ck.hpp"

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
  auto full_path = find_resource(options.opacity_file()[0]);

#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  nc_open(full_path.c_str(), NC_NETCDF4, &fileid);

  nc_inq_dimid(fileid, "Wavenumber", &dimid);
  nc_inq_dimlen(fileid, dimid, kshape);
  nc_inq_dimid(fileid, "Pressure", &dimid);
  nc_inq_dimlen(fileid, dimid, kshape + 1);
  nc_inq_dimid(fileid, "TempGrid", &dimid);
  nc_inq_dimlen(fileid, dimid, kshape + 2);

  kaxis = torch::empty({kshape[0] + kshape[1] + kshape[2]}, torch::kFloat64);

  // (wavenumber)g-grid
  nc_inq_varid(fileid, "Wavenumber", &varid);
  nc_get_var_double(fileid, varid, kaxis.data_ptr<double>());

  // pressure grid
  err = nc_inq_varid(fileid, "Pressure", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kaxis.data_ptr<double>() + kshape[0]);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // temperature grid
  err = nc_inq_varid(fileid, "TempGrid", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid,
                          kaxis.data_ptr<double>() + kshape[0] + kshape[1]);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // reference atmosphere
  double* temp = new double[kshape[1]];
  nc_inq_varid(fileid, "Temperature", &varid);
  err = nc_get_var_double(fileid, varid, temp);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  krefatm = torch::empty({2, kshape[1]}, torch::kFloat64);
  for (int i = 0; i < kshape[1]; i++) {
    krefatm[IPR][i] = kaxis[kshape[0] + i];
    krefatm[IDN][i] = temp[i];
  }
  delete[] temp;

  // ck data
  kdata = torch::empty({kshape[0], kshape[1], kshape[2]}, torch::kFloat64);
  auto name = options.species_names()[options.species_ids()[0]];

  err = nc_inq_varid(fileid, name.c_str(), &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kdata.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // ck weights
  kweight = torch::empty({kshape[0]}, torch::kFloat64);

  err = nc_inq_varid(fileid, "weights", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kweight.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  nc_close(fileid);
#endif
}

torch::Tensor RFMCKImpl::forward(torch::Tensor xfrac, torch::Tensor pres,
                                 torch::Tensor temp) {
  int nwave = kshape[0];
  int ncol = xfrac.size(0);
  int nlyr = xfrac.size(1);

  // get temperature anomaly
  auto tempa = temp - get_reftemp(krefatm, pres.log());

  auto out = torch::zeros({nwave, ncol, nlyr}, xfrac.options());
  auto dims = torch::tensor(
      {kshape[1], kshape[2]},
      torch::TensorOptions().dtype(torch::kInt64).device(xfrac.device()));
  auto axis = torch::empty({ncol, nlyr, 2}, torch::kFloat64);

  // first axis is pressure, second is temperature anomaly
  axis.select(2, IPR).copy_(pres);
  axis.select(2, IDN).copy_(tempa);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/0)
                  .add_output(out)
                  .add_owned_const_input(
                      axis.unsqueeze(0).expand({nwave, ncol, nlyr, 2}))
                  .build();

  if (xfrac.is_cpu()) {
    call_interpn_cpu<1>(iter, kdata, axis, dims, /*nval=*/1);
  } else if (xfrac.is_cuda()) {
    // call_interpn_cuda<1>(iter, kdata, kwave, dims, 1);
  } else {
    throw std::runtime_error("Unsupported device");
  }

  auto dens = pres / (Constants::Rgas * temp);
  return 1.E-3 * out.exp() * dens * xfrac;  // ln(m*2/kmol) -> 1/m
}

torch::Tensor get_reftemp(torch::Tensor refatm, torch::Tensor lnp) {
  int ncol = lnp.size(0);
  int nlyr = lnp.size(1);

  auto out = torch::zeros({ncol, nlyr}, xfrac.options());
  auto dims = torch::tensor(
      {lyr}, torch::TensorOptions().dtype(torch::kInt64).device(lnp.device()));

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(out.sizes(), /*squash_dims=*/1)
          .add_output(out)
          .add_owned_const_input(refatm[IPR].unsqueeze(0).expand({ncol, -1}))
          .build();

  if (xfrac.is_cpu()) {
    call_interpn_cpu<1>(iter, krefatm[ITM], refatm[IPR], dims, /*nval=*/1);
  } else if (xfrac.is_cuda()) {
    // call_interpn_cuda<1>(iter, kdata, kwave, dims, 1);
  } else {
    throw std::runtime_error("Unsupported device");
  }

  return out;
}

}  // namespace harp
