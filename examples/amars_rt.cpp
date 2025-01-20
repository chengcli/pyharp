// harp
#include <opacity/h2so4_simple.hpp>
#include <opacity/s8_fuller.hpp>
#include <rtsolver/rtsolver.hpp>

harp::DisortOptions disort_options(int nwave, int ncol, int nlyr) {
  harp::DisortOptions op;

  op.header("running amars RT");
  op.flags(
      "lamber,quiet,onlyfl,"
      "intensity_correction,old_intensity_correction,"
      "print-input,print-phase-function");

  op.nwave(nwave);
  op.ncol(ncol);

  op.ds().nlyr = nlyr;
  op.ds().nstr = 8;
  op.ds().nmom = 8;

  op.ds().nphi = 1;
  op.ds().ntau = 1;
  op.ds().numu = 1;

  return op;
}

/*void set_disort_bc(harp::Disort &disort, int nwave, int ncol) {
  for (int i = 0; i < nwave; ++i)
    for (int j = 0; j < ncol; ++j) {
      disort->ds(i, j).bc.umu0 = 0.1;
      disort->ds(i, j).bc.phi0 = 0.0;
      disort->ds(i, j).bc.albedo = 0.0;
      disort->ds(i, j).bc.fluor = 0.0;
      disort->ds(i, j).bc.fisot = 0.0;
    }
}*/

torch::Tensor atm_concentration(int ncol, int nlyr, int nspecies) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  return conc;
}

torch::Tensor short_wavenumber_grid(int nwave) {
  int wmin = 10000;
  int wmax = 100000;
  return torch::linspace(wmin, wmax, nwave, torch::kFloat64);
}

torch::Tensor short_toa_flux(int nwave, int ncol) {
  return torch::ones({nwave, ncol}, torch::kFloat64);
}

int main(int argc, char **argv) {
  int nwave = 10;
  int ncol = 1;
  int nlyr = 10;
  int nspecies = 2;

  harp::Disort disort(disort_options(nwave, ncol, nlyr));
  // set_disort_bc(disort, nwave, ncol);

  harp::S8Fuller s8(harp::S8RTOptions().species_id(0));
  harp::H2SO4Simple h2so4(harp::H2SO4RTOptions().species_id(1));

  auto wave = short_wavenumber_grid(nwave);
  auto conc = atm_concentration(ncol, nlyr, nspecies);

  auto prop1 = s8->forward(wave, conc);
  auto prop2 = h2so4->forward(wave, conc);

  auto prop = prop1 + prop2;

  // mean single scattering albedo
  prop.select(3, 1) /= prop.select(3, 0);

  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = short_toa_flux(nwave, ncol);
  bc["umu0"] = 0.1 * torch::ones({nwave, ncol}, torch::kFloat64);

  auto result = disort->forward(prop, bc);
  std::cout << result << std::endl;
}
