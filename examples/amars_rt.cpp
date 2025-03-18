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
  double btemp0 = 210;
  double solar_temp = 5772;
  double lum_scale = 0.7;

  /// ----- read atmosphere data -----  ///

  auto aero_ptx = harp::read_data_tensor("aerosol_output_data.txt");
  auto aero_p = (aero_ptx.select(1, 0) * 1e5);
  auto aero_t = aero_ptx.select(1, 1);
  auto aero_x = aero_ptx.narrow(1, 2, 2);

  // unit = [pa]
  auto new_P = harp::read_data_tensor("pVals.txt").squeeze(1);
  new_P *= 100.0;  // convert mbar to Pa

  // unit = [K]
  auto new_T = harp::read_data_tensor("TVals.txt").squeeze(1);
  auto dT_dt = torch::zeros_like(new_T);

  // unit = [mol/mol]
  auto new_X = harp::interpn({new_P.log()}, {aero_p.log()}, aero_x);

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

  // unit = [kg/m^3]
  auto new_rho = (new_P * mean_mol_weight) / (R * new_T);

  // unit = [m]
  auto dz = harp::calc_dz_hypsometric(new_P, new_T,
                                      torch::tensor({mean_mol_weight * g / R}));

  /// ----- done read atmosphere data -----  ///

  // configure input data for each radiation band
  std::map<std::string, torch::Tensor> atm, bc;
  atm["pres"] = new_P.unsqueeze(0).expand({ncol, nlyr});
  atm["temp"] = new_T.unsqueeze(0).expand({ncol, nlyr});
  atm["rho"] = new_rho.unsqueeze(0).expand({ncol, nlyr});

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
      bc[name + "/umu0"] = 0.707 * torch::ones({ncol}, torch::kFloat64);
    } else {  // longwave
      band.disort().wave_lower(std::vector<double>(nwave, wmin));
      band.disort().wave_upper(std::vector<double>(nwave, wmax));
      bc[name + "/albedo"] = 0.0 * torch::ones({nwave, ncol}, torch::kFloat64);
    }
  }
  bc["btemp"] = btemp0 * torch::ones({ncol}, torch::kFloat64);
  bc["ttemp"] = torch::zeros({ncol}, torch::kFloat64);

  // print radiation options and construct radiation model
  harp::Radiation rad(op);
  auto netflux = rad->forward(conc, dz, &bc, &atm);

  int t_lim = 10000;
  int print_freq = 500;
  double tstep = 86400 / 4.;
  double cSurf =
      200000;  // thermal inertia of the surface, assuming half ocean half land
  double current_time = 0;

  for (int t_ind = 0; t_ind < t_lim; ++t_ind) {
    auto surf_forcing =
        harp::shared["radiation/downward_flux"] - 5.67e-8 * bc["btemp"].pow(4);
    bc["btemp"] += surf_forcing * (tstep / cSurf);

    dT_dt = torch::zeros_like(atm["temp"]);
    auto dT_dt = -1. / (atm["rho"] * cp) *
                 (netflux.narrow(-1, 1, nlyr) - netflux.narrow(-1, 0, nlyr)) /
                 dz;

    atm["temp"] += dT_dt * tstep;
    atm["temp"].clamp_(20, 1000);

    conc.select(-1, 0) = (atm_data.data["CO2 [ppmv]"] * 1e-6) *
                         (atm["pres"] / (R * atm["temp"]));
    conc.select(-1, 1) = (atm_data.data["H2O [ppmv]"] * 1e-6) *
                         (atm["pres"] / (R * atm["temp"]));
    conc.select(-1, 2) = (atm_data.data["SO2 [ppmv]"] * 1e-6) *
                         (atm["pres"] / (R * atm["temp"]));

    // aerosols
    conc.narrow(-1, 3, new_X.size(-1)) = aero_scale * new_X.unsqueeze(0) *
                                         atm["pres"].unsqueeze(-1) /
                                         (R * atm["temp"].unsqueeze(-1));

    // unit = [kg/m^3]
    atm["rho"] = (atm["pres"] * mean_mol_weight) / (R * atm["temp"]);

    // unit = [m]
    dz = harp::calc_dz_hypsometric(atm["pres"], atm["temp"],
                                   torch::tensor({mean_mol_weight * g / R}));

    if (t_ind % print_freq == 0) {
      std::ostringstream filename;
      filename << "tp_result" << t_ind << ".txt";

      // Open the file and write the data
      std::ofstream outputFile3(filename.str());
      outputFile3 << "#p[Pa] T[K] netF[w/m^2] dT/dt [K/s]" << std::endl;
      for (int k = 0; k < nlyr; ++k) {
        outputFile3 << atm["pres"][0][k].item<double>() << " "
                    << atm["temp"][0][k].item<double>() << " "
                    << atm["temp"][0][k].item<double>() << " "
                    << dT_dt[0][k].item<double>() << std::endl;
      }
      outputFile3.close();
    }

    netflux = rad->forward(conc, dz, &bc, &atm);
  }
}
