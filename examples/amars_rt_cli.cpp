// torch
#include <torch/torch.h>

// harp
#include <radiation/disort_config.hpp>
#include <radiation/radiation.hpp>
#include <radiation/radiation_formatter.hpp>

int main(int argc, char** argv) {
  // parameters of the computational grid
  int ncol = 1;
  int nlyr = 40;
  int nstr = 8;

  // parameters of the amars model
  double g = 3.711;
  double mean_mol_weight = 0.044;  // CO2
  double R = 8.314472;
  double cp = 844;  // J/(kg K) for CO2
  double surf_sw_albedo = 0.3;
  // double aero_scale = 1e-6;
  double aero_scale = 1;
  double sr_sun = 2.92842e-5;  // angular size of the sun at mars
  double btemp = 210;

  // configure wave grid and parameters for each band
  std::map<std::string, torch::Tensor> atm, bc;
  atm["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  atm["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  for (int i = 0; i < nlyr; ++i) {
    atm["pres"][0][i] = new_p[i];
    atm["temp"][0][i] = new_T[i];
  }
  bc["umu0"] = 0.707 * torch::ones({nwave, ncol}, torch::kFloat64);
  bc["btemp"] = btemp * torch::ones({nwave, ncol}, torch::kFloat64);

  // read radiation configuration from yaml file
  auto op = harp::RadiationOptions::from_yaml("amars-ck.yaml");
  for (auto& [name, band] : op.band_options()) {
    int nwave = name == "SW" ? 500 : band.get_num_waves();

    auto wmin = band.disort().wave_lower()[0];
    auto wmax = band.disort().wave_upper()[0];

    band.disort(harp::disort_config(nwave, ncol, nlyr, nstr));

    if (name == "SW") {  // shortwave
      auto wave = torch::linspace(wmin, wmax, nwave, torch::kFloat64);
      atm[name + "/wavenumber"] = wave;
      bc[name + "/fbeam"] =
          lum_scale * sr_sun * bb_flux(wave, solar_temp, ncol);
      bc[name + "/albedo"] =
          surf_sw_albedo * torch::ones({nwave, ncol}, torch::kFloat64);
    } else {  // longwave
      band.disort().wave_lower(std::vector<double>(nwave, wmin));
      band.disort().wave_upper(std::vector<double>(nwave, wmax));
      bc[name + "/albedo"] = 0.0 * torch::ones({nwave, ncol}, torch::kFloat64);
    }
  }

  // print radiation options and construct radiation model
  std::cout << "rad op = " << fmt::format("{}", op) << std::endl;
  harp::Radiation rad(op);

  auto flux = rad->forward(conc, dz, &bc, &kwargs);
}
