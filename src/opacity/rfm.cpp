// base
#include <configure.h>

// harp
#include <math/interpolation.hpp>
#include <utils/find_resource.hpp>

#include "rfm.hpp"

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {

RFMImpl::RFMImpl(AttenuatorOptions const& options_) : options(options_) {
  TORCH_CHECK(options.opacity_files().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(options.type().empty() || (options.type() == "rfm"),
              "Mismatch type: ", options.type());

  reset();
}

void RFMImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);

#ifdef NETCDFOUTPUT
  int fileid, dimid, varid, err;
  nc_open(full_path.c_str(), NC_NETCDF4, &fileid);

  err = nc_inq_dimid(fileid, "Wavenumber", &dimid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_dimlen(fileid, dimid, kshape);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_dimid(fileid, "Pressure", &dimid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_dimlen(fileid, dimid, kshape + 1);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_dimid(fileid, "TempGrid", &dimid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_inq_dimlen(fileid, dimid, kshape + 2);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  kaxis = torch::empty({(int)kshape[0] + (int)kshape[1] + (int)kshape[2]},
                       torch::kFloat64);

  // wave grid
  err = nc_inq_varid(fileid, "Wavenumber", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kaxis.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // pressure grid
  err = nc_inq_varid(fileid, "Pressure", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kaxis.data_ptr<double>() + kshape[0]);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // change pressure to ln-pressure
  kaxis.slice(0, kshape[0], kshape[0] + kshape[1]).log_();

  // temperature grid
  err = nc_inq_varid(fileid, "TempGrid", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid,
                          kaxis.data_ptr<double>() + kshape[0] + kshape[1]);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // reference atmosphere
  double* temp = new double[kshape[1]];
  err = nc_inq_varid(fileid, "Temperature", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, temp);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  krefatm = torch::empty({2, (int)kshape[1]}, torch::kFloat64);
  for (int i = 0; i < kshape[1]; i++) {
    krefatm[IPR][i] = kaxis[kshape[0] + i];
    krefatm[ITM][i] = temp[i];
  }
  delete[] temp;

  // data
  kdata = torch::empty({(int)kshape[0], (int)kshape[1], (int)kshape[2]},
                       torch::kFloat64);
  auto name = options.species_names()[options.species_ids()[0]];

  err = nc_inq_varid(fileid, name.c_str(), &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kdata.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  nc_close(fileid);
#endif

  // register all buffers
  register_buffer("kaxis", kaxis);
  register_buffer("kdata", kdata);
  register_buffer("krefatm", krefatm);
}

torch::Tensor RFMImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int nwave = kshape[0];
  int ncol = conc.size(0);
  int nlyr = conc.size(1);
  constexpr int nprop = 1;

  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  // get temperature anomaly
  auto lnp = pres.log();
  auto tempa = temp - get_reftemp(lnp, krefatm[IPR], krefatm[ITM]);

  auto out = torch::zeros({nwave, ncol, nlyr, nprop}, conc.options());
  auto dims = torch::tensor(
      {(int)kshape[0], (int)kshape[1], (int)kshape[2]},
      torch::TensorOptions().dtype(torch::kInt64).device(conc.device()));
  auto coord = torch::empty({nwave, ncol, nlyr, 3}, torch::kFloat64);

  // first coord is wave, second log-pressure, third temperature anomaly
  coord.select(3, 0).copy_(kaxis.slice(0, 0, nwave).view({-1, 1, 1}));
  coord.select(3, 1).copy_(lnp);
  coord.select(3, 2).copy_(tempa);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/3)
                  .add_output(out)
                  .add_input(coord)
                  .build();

  if (conc.is_cpu()) {
    call_interpn_cpu<nprop>(iter, kdata, kaxis, dims, /*nval=*/nprop);
  } else if (conc.is_cuda()) {
    // call_interpn_cuda<1>(iter, kdata, kwave, dims, 1);
  } else {
    TORCH_CHECK(false, "Unsupported device");
  }

  // ln(m*2/kmol) -> 1/m
  return 1.E-3 * out.exp() *
         conc.select(2, options.species_ids()[0]).unsqueeze(0).unsqueeze(-1);
}

torch::Tensor get_reftemp(torch::Tensor lnp, torch::Tensor klnp,
                          torch::Tensor ktemp) {
  int ncol = lnp.size(0);
  int nlyr = lnp.size(1);

  auto out = torch::zeros({ncol, nlyr}, lnp.options());
  auto dims = torch::tensor(
      {nlyr}, torch::TensorOptions().dtype(torch::kInt64).device(lnp.device()));

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes())
                  .add_output(out)
                  .add_input(lnp)
                  .build();

  if (lnp.is_cpu()) {
    call_interpn_cpu<1>(iter, ktemp, klnp, dims, /*nval=*/1);
  } else if (lnp.is_cuda()) {
    // call_interpn_cuda<1>(iter, kdata, kwave, dims, 1);
  } else {
    TORCH_CHECK(false, "Unsupported device");
  }

  return out;
}

}  // namespace harp
