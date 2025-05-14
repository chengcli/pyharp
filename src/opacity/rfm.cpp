// base
#include <configure.h>

// harp
#include <harp/math/interpolation.hpp>
#include <harp/utils/find_resource.hpp>

#include "rfm.hpp"

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

namespace harp {

extern std::vector<std::string> species_names;

RFMImpl::RFMImpl(AttenuatorOptions const& options_) : options(options_) {
  TORCH_CHECK(options.opacity_files().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(
      options.type().empty() || (options.type().compare(0, 3, "rfm") == 0),
      "Mismatch opacity type: ", options.type());

  reset();
}

void RFMImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);

  // data table shape (nwave, npres, ntemp)
  size_t kshape[3];

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

  // wavenumber grid
  kwave = torch::empty({(int)kshape[0]}, torch::kFloat64);

  // pressure grid
  klnp = torch::empty({(int)kshape[1]}, torch::kFloat64);

  // temperatur grid
  ktempa = torch::empty({(int)kshape[2]}, torch::kFloat64);

  // wave grid
  err = nc_inq_varid(fileid, "Wavenumber", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kwave.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // pressure grid
  err = nc_inq_varid(fileid, "Pressure", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, klnp.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));
  // FIXME: dirty fix pressure unit
  // klnp /= 100.;

  // change pressure to ln-pressure
  klnp.log_();

  // temperature grid
  err = nc_inq_varid(fileid, "TempGrid", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, ktempa.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // reference temperature
  kreftem = torch::empty({(int)kshape[1], 1}, torch::kFloat64);
  err = nc_inq_varid(fileid, "Temperature", &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kreftem.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  // data
  kdata = torch::empty({(int)kshape[0], (int)kshape[1], (int)kshape[2], 1},
                       torch::kFloat64);
  auto name = species_names[options.species_ids()[0]];

  err = nc_inq_varid(fileid, name.c_str(), &varid);
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  err = nc_get_var_double(fileid, varid, kdata.data_ptr<double>());
  TORCH_CHECK(err == NC_NOERR, nc_strerror(err));

  nc_close(fileid);
#endif

  // register all buffers
  register_buffer("kwave", kwave);
  register_buffer("klnp", klnp);
  register_buffer("ktempa", ktempa);
  register_buffer("kdata", kdata);
  register_buffer("kreftem", kreftem);
}

torch::Tensor RFMImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int nwave = kwave.size(0);
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

  // get temperature anomaly
  auto lnp = pres.log();
  auto tempa = temp - interpn({lnp}, {klnp}, kreftem).squeeze(-1);

  // interpolate data
  auto wave = kwave.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  lnp = lnp.unsqueeze(0).expand({nwave, ncol, nlyr});
  tempa = tempa.unsqueeze(0).expand({nwave, ncol, nlyr});

  auto out = interpn({wave, lnp, tempa}, {kwave, klnp, ktempa}, kdata);

  // Check species id in range
  TORCH_CHECK(
      options.species_ids()[0] >= 0 && options.species_ids()[0] < conc.size(2),
      "Invalid species_id: ", options.species_ids()[0]);

  // ln(m*2/kmol) -> 1/m
  return 1.E-3 * out.exp() *
         conc.select(-1, options.species_ids()[0]).unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
