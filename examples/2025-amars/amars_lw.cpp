// torch
#include <torch/torch.h>

// harp
#include <harp/radiation/disort_config.hpp>
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_band.hpp>

// unit = [mol/m^3]
torch::Tensor atm_concentration(int ncol, int nlyr, int nspecies) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  return conc;
}

//! \brief Main function for testing AMARS longwave radiation
/*!
 * This is an example program that illustrates how to manually set up
 * opacity sources and run a radiative transfer calculation.
 * Normally, you would initialize the model from a yaml configuration file.
 * However, this example shows the internal mechanisms more clearly.
 */
int main(int argc, char** argv) {
  harp::species_names = {"CO2", "H2O"};
  harp::species_weights = {44.0e-3, 18.0e-3};

  auto op_co2 = harp::OpacityOptionsImpl::create();
  (*op_co2).type("rfm-ck").species_ids({0}).opacity_files({"amarsw-ck-B1.nc"});

  auto op_h2o = harp::OpacityOptionsImpl::create();
  (*op_h2o).type("rfm-ck").species_ids({1}).opacity_files({"amarsw-ck-B1.nc"});

  auto lw_op = harp::RadiationBandOptionsImpl::create();
  (*lw_op).name("lw").solver_name("disort").opacities({
      {"CO2", op_co2},
      {"H2O", op_h2o},
  });

  int nwave = op_co2->query_wavenumber().size();
  int ncol = 1;
  int nlyr = 40;
  double wmin = 1.;
  double wmax = 150.;

  lw_op->nwave(nwave);
  lw_op->ncol(ncol);
  lw_op->nlyr(nlyr);

  lw_op->disort() =
      harp::create_disort_config_lw(wmin, wmax, nwave, ncol, nlyr);
  lw_op->weight() = op_co2->query_weight();

  harp::RadiationBand lw(lw_op);

  auto conc = atm_concentration(ncol, nlyr, harp::species_names.size());

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64);
  bc["btemp"] = torch::ones({ncol}, torch::kFloat64) * 300.0;

  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 10.e5;
  atm["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 300.0;

  auto dz = torch::ones({nlyr}, torch::kFloat64);
  auto flux = lw->forward(conc, dz, &bc, &atm);
  std::cout << "spectrum shape = " << lw->spectrum.sizes() << std::endl;
  std::cout << "result = " << flux << std::endl;
}
