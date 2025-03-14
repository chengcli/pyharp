// C/C++
#include <fstream>
#include <iostream>

// torch
#include <torch/torch.h>

// harp
#include <math/interpolation.hpp>
#include <radiation/bbflux.hpp>
#include <radiation/calc_dz_hypsometric.hpp>
#include <radiation/disort_config.hpp>
#include <radiation/radiation.hpp>
#include <radiation/radiation_formatter.hpp>
#include <utils/read_data_tensor.hpp>

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

  // init some params for the loop
  double inc_sw_surf = 0;
  double inc_lw_surf = 0;
  double surf_forcing = 0;

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
      bc[name + "/umu0"] = 0.707 * torch::ones({nwave, ncol}, torch::kFloat64);
    } else {  // longwave
      band.disort().wave_lower(std::vector<double>(nwave, wmin));
      band.disort().wave_upper(std::vector<double>(nwave, wmax));
      bc[name + "/albedo"] = 0.0 * torch::ones({nwave, ncol}, torch::kFloat64);
      bc[name + "/btemp"] = btemp * torch::ones({nwave, ncol}, torch::kFloat64);
    }
  }

  // print radiation options and construct radiation model
  harp::Radiation rad(op);
  auto netflux = rad->forward(conc, dz, &bc, &atm);
  
  int t_lim = 10000; 
  int print_freq = 500;
  double tstep = 86400 / 4;  
  double cSurf = 200000;  // thermal inertia of the surface, assuming half ocean half land
  double current_time = 0;

  for (int t_ind = 0; t_ind < t_lim; ++t_ind) {
    inc_sw_surf = 0;
    inc_lw_surf = 0;
    for (auto& [name, band] : op.band_options()) {
      std::string name1 = "radiation/" + name + "/total_flux";
      if (name == "SW"){
        //std::cout << harp::shared[name1] << std::endl;
        //std::cout << harp::shared[name1][0][0][1] << std::endl;
        inc_sw_surf += harp::shared[name1][0][0][1].item<double>();
      }
      else{
        inc_lw_surf += harp::shared[name1][0][0][1].item<double>();

      }
    }
    surf_forcing = (1 - surf_sw_albedo) * inc_sw_surf + inc_lw_surf - 5.67e-8 * std::pow(btemp, 4);
    btemp += surf_forcing * (tstep / cSurf);


    for (auto& [name, band] : op.band_options()) {
      int nwave = name == "SW" ? 500 : band.ww().size();
      if (name != "SW"){
        bc[name + "/btemp"] = btemp * torch::ones({nwave, ncol}, torch::kFloat64);
      }
    }

    dT_dt = torch::zeros_like(new_T);
    for (int i = 0; i < nlyr; ++i) {
        dT_dt[i] += (-1 / (new_rho[i].item<double>() * cp)) * ((netflux[0][i + 1].item<double>() - netflux[0][i].item<double>()) / dz[i].item<double>());
    }
    
    new_T += dT_dt * tstep;
    for (int k = 0; k < nlyr; ++k) {
      if (new_T[k].item<double>() < 20) new_T[k] = 20;
    }
    atm["temp"] = new_T.unsqueeze(0).expand({ncol, nlyr});

    conc.select(-1, 0) = (atm_data.data["CO2 [ppmv]"] * 1e-6) * (new_P / (R * new_T));
    conc.select(-1, 1) = (atm_data.data["H2O [ppmv]"] * 1e-6) * (new_P / (R * new_T));
    conc.select(-1, 2) = (atm_data.data["SO2 [ppmv]"] * 1e-6) * (new_P / (R * new_T));



  //        conc[0][k][0] =
  //        (aero_scale * new_mr[1][k] * new_p[k]) /
  //        (R * new_T[k]);  // s8 comes second in the file that we read
                           // in. but we need it to be index 0 in conc
                           // bc of how it was initialized above

    // aerosols
    conc.narrow(-1, 3, new_X.size(-1)) =
        aero_scale * new_X * new_P.unsqueeze(1) / (R * new_T.unsqueeze(1));

    // unit = [kg/m^3]
    new_rho = (new_P * mean_mol_weight) / (R * new_T);

    // unit = [m]
    dz = harp::calc_dz_hypsometric(new_P, new_T,
                                        torch::tensor({mean_mol_weight * g / R}));


    if (t_ind % print_freq == 0) {
      std::ostringstream filename;
      filename << "tp_result" << t_ind << ".txt";

      // Open the file and write the data
      std::ofstream outputFile3(filename.str());
      outputFile3 << "#p[Pa] T[K] dT/dt [K/s]" << std::endl;
      for (int k = 0; k < nlyr; ++k) {
        outputFile3 << new_P[k].item<double>() << " " << new_T[k].item<double>() << " " << dT_dt[k].item<double>()
                    << std::endl;
      }
      outputFile3.close();
    }

    netflux = rad->forward(conc, dz, &bc, &atm);
  }
}
