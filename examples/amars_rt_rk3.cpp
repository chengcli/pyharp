// C/C++
#include <fstream>
#include <iostream>

// torch
#include <torch/torch.h>

// harp
#include <integrator/radiation_model.hpp>
#include <math/interpolation.hpp>
#include <radiation/bbflux.hpp>
#include <radiation/disort_config.hpp>
#include <radiation/radiation.hpp>
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
  int nstr = 4;

  // parameters of the amars model
  double surf_sw_albedo = 0.3;
  double sr_sun = 2.92842e-5;  // angular size of the sun at mars
  double btemp0 = 210;
  double solar_temp = 5772;
  double lum_scale = 0.7;

  /// ----- read atmosphere data -----  ///

  auto aero_ptx = harp::read_data_tensor("aerosol_output_data.txt");
  auto aero_p = (aero_ptx.select(1, 0) * 1e5);  // bar to pa
  auto aero_t = aero_ptx.select(1, 1);
  auto aero_x = aero_ptx.narrow(1, 2, 2);

  // unit = [pa]
  // log pressure grid from 500 mbar to 1 mbar
  auto new_P = torch::logspace(log10(500.0), log10(1.0), nlyr);
  new_P *= 100.0;  // convert mbar to Pa

  // unit = [K]
  // isothermal temperature profile at 200 K
  auto new_T = 200. * torch::ones({nlyr}, torch::kFloat64);

  // unit = [mol/mol]
  auto new_X = harp::interpn({new_P.log()}, {aero_p.log()}, aero_x);

  AtmosphericData atm_data = read_rfm_atm("rfm.atm");
  auto pre = atm_data.data["PRE [mb]"] * 100.0;
  auto tem = atm_data.data["TEM [K]"];

  // unit = [mol/mol]
  // mole fraction
  auto xfrac = torch::zeros({ncol, nlyr, 5}, torch::kFloat64);
  xfrac.select(-1, 0) =
      harp::interpn({new_P.log()}, {pre.log()},
                    atm_data.data["CO2 [ppmv]"].unsqueeze(-1) * 1e-6)
          .squeeze(-1);

  xfrac.select(-1, 1) =
      harp::interpn({new_P.log()}, {pre.log()},
                    atm_data.data["H2O [ppmv]"].unsqueeze(-1) * 1e-6)
          .squeeze(-1);

  xfrac.select(-1, 2) =
      harp::interpn({new_P.log()}, {pre.log()},
                    atm_data.data["SO2 [ppmv]"].unsqueeze(-1) * 1e-6)
          .squeeze(-1);

  // aerosols
  xfrac.narrow(-1, 3, new_X.size(-1)) = new_X;

  /// ----- done read atmosphere data -----  ///
  std::map<std::string, torch::Tensor> atm, bc;
  // set up model atmosphere
  atm["pres"] = new_P.unsqueeze(0).expand({ncol, nlyr});
  atm["temp"] = new_T.unsqueeze(0).expand({ncol, nlyr});

  // read radiation configuration from yaml file
  auto rad_op = harp::RadiationOptions::from_yaml("amars-ck.yaml");
  for (auto& [name, band] : rad_op.band_options()) {
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

  // parameters of the amars model
  harp::RadiationModelOptions model_op;
  model_op.ncol(ncol);
  model_op.nlyr(nlyr);
  model_op.grav(3.711);
  model_op.mean_mol_weight(0.044);  // CO2
  model_op.cp(844);                 // J/(kg K) for CO2
  model_op.aero_scale(1.0);
  model_op.cSurf(200000);  // J/(m^2 K) thermal intertia of the surface
  model_op.intg(harp::IntegratorOptions().type("rk3"));
  model_op.rad(rad_op);

  harp::RadiationModel model(model_op);

  int t_lim = 10000;
  double tstep = 86400 / 4.;
  int print_freq = 500;

  for (int t_ind = 0; t_ind < t_lim; ++t_ind) {
    for (int stage = 0; stage < model->pintg->stages.size(); ++stage) {
      model->forward(xfrac, atm, bc, tstep, stage);
    }

    if (t_ind % print_freq == 0) {
      std::ostringstream filename;
      filename << "tp_result" << t_ind << ".txt";

      // Open the file and write the data
      std::ofstream outputFile3(filename.str());
      outputFile3 << "#p[Pa] T[K] netF[w/m^2] dT/dt [K/s]" << std::endl;
      for (int k = 0; k < nlyr; ++k) {
        outputFile3 << atm["pres"][0][k].item<double>() << " "
                    << atm["temp"][0][k].item<double>() << " "
                    << harp::shared["result/netflux"][0][k].item<double>()
                    << " " << harp::shared["result/dT_atm"][0][k].item<double>()
                    << std::endl;
      }
      outputFile3.close();
    }
  }
}
