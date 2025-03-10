// torch
#include <torch/torch.h>

// harp
#include <radiation/disort_options_flux.hpp>
#include <radiation/radiation_band.hpp>

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

  harp::RadiationBandOptions lw_op;
  lw_op.name() = "lw";
  lw_op.solver_name() = "disort";
  lw_op.opacities() = {
      {"CO2",
       op.species_ids({0}).opacity_files({"amarsw-ck-B1.nc"}).type("rfm-ck")},
      {"H2O",
       op.species_ids({1}).opacity_files({"amarsw-ck-B1.nc"}).type("rfm-ck")},
  };
  lw_op.integration() = "weight";

  int nwave = lw_op.get_num_waves();
  lw_op.disort() = harp::disort_flux_lw(wmin, wmax, nwave, ncol, nlyr);
  harp::RadiationBand lw(lw_op);

  auto conc = atm_concentration(ncol, nlyr, op.species_names().size());

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64);
  bc["btemp"] = torch::ones({nwave, ncol}, torch::kFloat64) * 300.0;

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 10.e5;
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 300.0;

  auto dz = torch::ones({ncol, nlyr}, torch::kFloat64);
  auto flux = lw->forward(conc, dz, &bc, &kwargs);
  std::cout << "result = " << flux << std::endl;
  // std::cout << harp::shared.at("radiation/lw/spectra") << std::endl;
}
