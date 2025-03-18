// C/C++
#include <fstream>
#include <iostream>

// torch
#include <torch/torch.h>

// harp
#include <harp/math/interpolation.hpp>
#include <harp/radiation/bbflux.hpp>
#include <harp/radiation/calc_dz_hypsometric.hpp>
#include <harp/radiation/disort_config.hpp>
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_formatter.hpp>
#include <harp/utils/read_data_tensor.hpp>

struct AtmosphericData {
  int n_layers;
  std::map<std::string, torch::Tensor> data;
};

AtmosphericData read_rfm_atm(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  AtmosphericData atm_data;
  std::string line;

  // Read the number of layers
  std::getline(file, line);
  std::istringstream iss(line);
  iss >> atm_data.n_layers;

  // Read the data sections
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    // Read the name of the data section
    if (line[0] == '*') {
      std::string name = line.substr(1);  // Remove the '*' character

      // Read the data points
      std::getline(file, line);
      std::istringstream data_stream(line);
      std::vector<double> data_points(atm_data.n_layers);
      for (int i = 0; i < atm_data.n_layers; ++i) {
        data_stream >> data_points[i];
      }

      // Store the data points in the map
      atm_data.data[name] = torch::tensor(data_points, torch::kFloat64);
    }
  }

  file.close();
  return atm_data;
}

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
  double solar_temp = 5772;
  double lum_scale = 0.7;

  /// ----- read atmosphere data -----  ///

  auto aero_ptx = harp::read_data_tensor("aerosol_output_data.txt");
  auto aero_p = (aero_ptx.select(1, 0) * 1e5);
  auto aero_t = aero_ptx.select(1, 1);
  auto aero_x = aero_ptx.narrow(1, 2, 2);
  std::cout << "aero_p = " << aero_p << std::endl;
  std::cout << "aero_t = " << aero_t << std::endl;
  std::cout << "aero_x = " << aero_x.sizes() << std::endl;

  // unit = [pa]
  auto new_P = harp::read_data_tensor("pVals.txt").squeeze(1);
  new_P *= 100.0;  // convert mbar to Pa
  std::cout << "new_P = " << new_P << std::endl;

  // unit = [K]
  auto new_T = harp::read_data_tensor("TVals.txt").squeeze(1);
  std::cout << "new_T = " << new_T << std::endl;

  // unit = [mol/mol]
  auto new_X = harp::interpn({new_P.log()}, {aero_p.log()}, aero_x);
  std::cout << "new_X = " << new_X << std::endl;

  AtmosphericData atm_data = read_rfm_atm("rfm.atm");
  auto pre = atm_data.data["PRE [mb]"] * 100.0;
  auto tem = atm_data.data["TEM [K]"];

  // unit = [mol/m^3]
  auto conc = torch::zeros({ncol, nlyr, 5}, torch::kFloat64);

  conc.select(-1, 0) = (atm_data.data["CO2 [ppmv]"] * 1e-6) * (pre / (R * tem));
  conc.select(-1, 1) = (atm_data.data["H2O [ppmv]"] * 1e-6) * (pre / (R * tem));
  conc.select(-1, 2) = (atm_data.data["SO2 [ppmv]"] * 1e-6) * (pre / (R * tem));

  // aerosols
  conc.narrow(-1, 3, new_X.size(-1)) =
      aero_scale * new_X * new_P.unsqueeze(1) / (R * new_T.unsqueeze(1));
  std::cout << "conc = " << conc << std::endl;

  // unit = [kg/m^3]
  auto new_rho = (new_P * mean_mol_weight) / (R * new_T);
  std::cout << "new_rho = " << new_rho << std::endl;

  // unit = [m]
  auto dz = harp::calc_dz_hypsometric(new_P, new_T,
                                      torch::tensor({mean_mol_weight * g / R}));
  std::cout << "dz = " << dz << std::endl;

  /// ----- done read atmosphere data -----  ///

  // configure input data for each radiation band
  std::map<std::string, torch::Tensor> atm, bc;
  atm["pres"] = new_P.unsqueeze(0).expand({ncol, nlyr});
  atm["temp"] = new_T.unsqueeze(0).expand({ncol, nlyr});

  // read radiation configuration from yaml file
  auto op = harp::RadiationOptions::from_yaml("amars-ck.yaml");
  for (auto& [name, band] : op.band_options()) {
    // query weights from opacity, only valid for longwave
    // shortwave values are defined separately
    band.ww() = band.query_weights();
    int nwave = name == "SW" ? 500 : band.ww().size();

    auto wmin = band.disort().wave_lower()[0];
    auto wmax = band.disort().wave_upper()[0];

    harp::disort_config(&band.disort(), nwave, ncol, nlyr, nstr);
    std::cout << "flags = " << band.disort().flags() << std::endl;

    if (name == "SW") {  // shortwave
      band.ww().resize(nwave);
      for (int i = 0; i < nwave; ++i) {
        band.ww()[i] = (wmax - wmin) * i / (nwave - 1) + wmin;
      }
      auto wave = torch::tensor(band.ww(), torch::kFloat64);
      bc[name + "/fbeam"] =
          lum_scale * sr_sun * harp::bbflux_wavenumber(wave, solar_temp, ncol);
      bc[name + "/albedo"] =
          surf_sw_albedo * torch::ones({nwave, ncol}, torch::kFloat64);
      bc[name + "/umu0"] = 0.707 * torch::ones({nwave, ncol}, torch::kFloat64);
    } else {  // longwave
      band.disort().wave_lower(std::vector<double>(nwave, wmin));
      band.disort().wave_upper(std::vector<double>(nwave, wmax));
      bc[name + "/albedo"] = 0.0 * torch::ones({nwave, ncol}, torch::kFloat64);
    }
  }
  bc["btemp"] = btemp * torch::ones({ncol}, torch::kFloat64);
  bc["ttemp"] = torch::zeros({ncol}, torch::kFloat64);

  // print radiation options and construct radiation model
  std::cout << "rad op = " << fmt::format("{}", op) << std::endl;
  harp::Radiation rad(op);
  auto netflux = rad->forward(conc, dz, &bc, &atm);
  std::cout << "net flux = " << netflux << std::endl;
  std::cout << "downward flux = " << harp::shared["radiation/downward_flux"]
            << std::endl;
  std::cout << "upward flux = " << harp::shared["radiation/upward_flux"]
            << std::endl;
}
