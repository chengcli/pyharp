// torch
#include <torch/torch.h>

// disort
#include <disort/disort.hpp>

// harp
#include <opacity/rfm.hpp>
#include <utils/layer2level.hpp>

// unit = [mol/m^3]
torch::Tensor atm_concentration(int ncol, int nlyr, int nspecies) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  return conc;
}

disort::DisortOptions disort_options_lw(double wmin, double wmax, int nwave,
                                        int ncol, int nlyr) {
  disort::DisortOptions op;

  op.header("running amars lw");
  op.flags(
      "lamber,quiet,onlyfl,planck,"
      "intensity_correction,old_intensity_correction,"
      "print-input,print-phase-function");

  op.nwave(nwave);
  op.ncol(ncol);
  op.wave_lower(std::vector<double>(nwave, wmin));
  op.wave_upper(std::vector<double>(nwave, wmax));

  op.ds().nlyr = nlyr;
  op.ds().nstr = 8;
  op.ds().nmom = 8;

  return op;
}

int main(int argc, char** argv) {
  harp::AttenuatorOptions op;
  op.species_names({"CO2", "H2O"});
  op.species_weights({44.0e-3, 18.0e-3});

  op.species_ids({0}).opacity_files({"amarsw-ck-B1.nc"});
  harp::RFM co2(op);

  op.species_ids({1}).opacity_files({"amarsw-ck-B1.nc"});
  harp::RFM h2o(op);

  int nwave = co2->kdata.size(0);
  int ncol = 1;
  int nlyr = 40;
  int nspecies = 2;

  double wmin = 1.;
  double wmax = 150.;
  auto conc = atm_concentration(ncol, nlyr, nspecies);

  disort::Disort disort(disort_options_lw(wmin, wmax, nwave, ncol, nlyr));

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 10.e5;
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 300.0;

  auto prop1 = co2->forward(conc, kwargs);
  auto prop2 = h2o->forward(conc, kwargs);

  auto prop = prop1 + prop2;
  std::cout << "prop = " << prop << std::endl;

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64);

  std::cout << "tem = " << kwargs["temp"] << std::endl;
  auto temf = harp::layer2level(kwargs["temp"], harp::Layer2LevelOptions());
  std::cout << "tempf = " << temf << std::endl;

  // assuming dz = 1
  auto result = disort->forward(prop, &bc, temf);
  std::cout << "result = " << result << std::endl;
}
