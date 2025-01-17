// harp
#include <opacity/h2so4_simple.hpp>
#include <opacity/s8_fuller.hpp>
#include <rtsolver/rtsolver.hpp>

harp::DisortOptions disort_options() {
  harp::DisortOptions op;

  op.header("running disort example");
  op.flags(
      "lamber,quiet,onlyfl,"
      "intensity_correction,old_intensity_correction,"
      "print-input,print-phase-function");

  op.ds().nlyr = 1;
  op.ds().nstr = 16;
  op.ds().nmom = 16;

  op.ds().nphi = 1;
  op.ds().ntau = 1;
  op.ds().numu = 1;

  return op;
}

void set_disort_bc(harp::Disort &disort) {
  disort->ds().bc.umu0 = 0.1;
  disort->ds().bc.phi0 = 0.0;
  disort->ds().bc.albedo = 0.0;
  disort->ds().bc.fluor = 0.0;
  disort->ds().bc.fisot = 0.0;
}

torch::Tensor atm_concentration() {
  int ncol = 1;
  int nlyr = 1;
  int nspecies = 2;
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  return conc;
}

torch::Tensor short_wavenumber_grid() {
  int wmin = 1000;
  int wmax = 10000;
  int nwave = 100;
  return torch::linspace(wmin, wmax, nwave, torch::kFloat64);
}

torch::Tensor short_toa_flux() {
  int nwave = 100;
  int ncol = 1;
  return torch::ones({nwave, ncol}, torch::kFloat64);
}

int main(int argc, char **argv) {
  harp::Disort disort(disort_options());
  set_disort_bc(disort);

  harp::S8Fuller s8(harp::S8RTOptions().species_id(0));
  harp::H2SO4Simple h2so4(harp::H2SO4RTOptions().species_id(1));

  auto wave = short_wavenumber_grid();
  auto conc = atm_concentration();

  auto prop1 = s8->forward(wave, conc);
  auto prop2 = h2so4->forward(wave, conc);

  auto prop = prop1 + prop2;
  auto ftoa = short_toa_flux();
  auto result = disort->forward(prop, ftoa);

  std::cout << result << std::endl;
}
