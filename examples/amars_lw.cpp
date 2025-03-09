// torch
#include <torch/torch.h>

// harp
#include <radiation/disort_options_flux.hpp>
#include <radiation/radiation_band.hpp>
#include <utils/read_weights.hpp>

// unit = [mol/m^3]
torch::Tensor atm_concentration(int ncol, int nlyr, int nspecies) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  return conc;
}

int main(int argc, char** argv) {
  harp::AttenuatorOptions op;
  op.species_names({"CO2", "H2O"});
  op.species_weights({44.0e-3, 18.0e-3});

  int ncol = 1;
  int nlyr = 40;

  double wmin = 1.;
  double wmax = 150.;

  RadiationBandOptions lw_op;
  lw_op.name() = "lw";
  lw_op.solver_name() = "disort";
  lw_op.opacity() = {
      {"CO2",
       op.species_ids({0}).opacity_files({"amarsw-ck-B1.nc"}).type("rfm")},
      {"H2O",
       op.species_ids({1}).opacity_files({"amarsw-ck-B1.nc"}).type("rfm")},
  };
  lw_op.disort() = disort_flux_lw(wmin, wmax, nwave, ncol, nlyr);
  RadiationBand lw(lw_op);

  auto conc = atm_concentration(ncol, nlyr, op.species_ids().size());

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64);
  bc["btemp"] = torch::ones({nwave, ncol}, torch::kFloat64) * 300.0;

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 10.e5;
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 300.0;

  auto dz = torch::ones({ncol, nlyr}, torch::kFloat64);
  auto flux = lw->forward(prop, dz, &bc, &kwargs);
  std::cout << "flux = " << flux << std::endl;

  // ck weights
  auto weights = harp::read_weights_rfm("amarsw-ck-B1.nc");
  std::cout << "weights = " << weights << std::endl;

  // band flux
  auto bflx = (flux * weights.view({-1, 1, 1, 1})).sum(0);
  std::cout << "bflx = " << bflx << std::endl;
}
