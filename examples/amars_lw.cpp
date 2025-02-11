#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>

// torch
#include <torch/torch.h>

// disort
#include <disort/disort.hpp>

// harp
#include <opacity/rfm.hpp>
#include <utils/layer2level.hpp>
#include <utils/read_weights.hpp>
#include <math/interpn.h>

struct AtmosphericData {
    int n_layers;
    std::map<std::string, std::vector<double>> data;
};

// unit = [mol/m^3]
torch::Tensor atm_concentration(int ncol, int nlyr, int nspecies, AtmosphericData atm_data) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  double pre = 0;
  double tem = 0;
  double R = 8.314472;
  for (int i = 0; i < nlyr; ++i) {
    pre = atm_data.data["PRE [mb]"][i] * 100.0;
    tem = atm_data.data["TEM [K]"][i];
    conc[0][i][0] = (atm_data.data["CO2 [ppmv]"][i] * 1e-6) * (pre / (R * tem));
    conc[0][i][1] = atm_data.data["H2O [ppmv]"][i] * 1e-6 * (pre / (R * tem));
  }
  return conc;
}

disort::DisortOptions disort_options_lw(double wmin, double wmax, int nwave,
                                        int ncol, int nlyr) {
  disort::DisortOptions op;

  op.header("running amars lw");
  op.flags(
      "lamber,quiet,onlyfl,planck,"
      "intensity_correction,old_intensity_correction,"
      "print-input,print-phase-function,print-fluxes");

  op.nwave(nwave);
  op.ncol(ncol);
  op.wave_lower(std::vector<double>(nwave, wmin));
  op.wave_upper(std::vector<double>(nwave, wmax));

  op.ds().nlyr = nlyr;
  op.ds().nstr = 8;
  op.ds().nmom = 8;

  return op;
}

AtmosphericData read_atmospheric_data(const std::string& filename) {
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
            std::string name = line.substr(1); // Remove the '*' character

            // Read the data points
            std::getline(file, line);
            std::istringstream data_stream(line);
            std::vector<double> data_points(atm_data.n_layers);
            for (int i = 0; i < atm_data.n_layers; ++i) {
                data_stream >> data_points[i];
            }

            // Store the data points in the map
            atm_data.data[name] = data_points;
        }
    }

    file.close();
    return atm_data;
}

torch::Tensor calc_dz(int nlyr, torch::Tensor prop, AtmosphericData atm_data) {
  auto dz = torch::ones({nlyr, 1}, prop.options());  // prop: (nwave, ncol, nlyr, nprop)
  double z_i = 0;
  double z_i_plus_1 = 0;

  dz[0][0] = atm_data.data["HGT [km]"][0] * 1000;
  for (int i = 1; i < nlyr - 1; ++i) {
    z_i = atm_data.data["HGT [km]"][i];
    z_i_plus_1 = atm_data.data["HGT [km]"][i + 1];
    dz[i][0] *= (z_i_plus_1 - z_i) * 1000;
  }
  dz[nlyr - 1][0] *= (z_i_plus_1 - z_i) * 1000;
  std::cout << "dz = " << dz << std::endl;
  return dz;
}

torch::Tensor calc_flux_1band(int ncol, int nspecies, double wmin, double wmax, AtmosphericData atm_data, std::string filename, int idx) {
  harp::AttenuatorOptions op;
  op.species_names({"CO2", "H2O"});
  op.species_weights({44.0e-3, 18.0e-3});

  op.species_ids({0}).opacity_files({filename});
  harp::RFM co2(op);

  op.species_ids({1}).opacity_files({filename});
  harp::RFM h2o(op);

  int nwave = co2->kdata.size(0);
  int nlyr = atm_data.n_layers;

  auto conc = atm_concentration(ncol, nlyr, nspecies, atm_data);

  disort::Disort disort(disort_options_lw(wmin, wmax, nwave, ncol, nlyr));

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  for (int i = 0; i < nlyr; ++i) {
    kwargs["pres"][0][i] = atm_data.data["PRE [mb]"][i] * 100.0;
    kwargs["temp"][0][i] = atm_data.data["TEM [K]"][i];
  }

  torch::Tensor prop;
  if (idx == 0) prop = co2->forward(conc, kwargs);
  if (idx == 1) prop = h2o->forward(conc, kwargs);
  auto dz = calc_dz(nlyr, prop, atm_data);

  prop *= dz;
  //std::cout << "prop shape = " << prop.sizes() << std::endl;

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64) * 0.0;
  bc["btemp"] = torch::ones({nwave, ncol}, torch::kFloat64) * 200.0;

  auto temf = harp::layer2level(kwargs["temp"], harp::Layer2LevelOptions());
  auto flux = disort->forward(prop, &bc, temf);
  auto weights = harp::read_weights_rfm(filename);

  return (flux * weights.view({-1, 1, 1, 1})).sum(0);
}

int main(int argc, char** argv) {
  double g = 3.711;
  AtmosphericData atm_data = read_atmospheric_data("/home/cometz/Desktop/rce/build-rt/bin/rfm.atm");
  int nlyr = atm_data.n_layers;
  int ncol = 1;
  int nspecies = 2;
  //CO2 is idx 0, H2O is idx 1
  auto tot_flux = calc_flux_1band(ncol, nspecies, 1., 150., atm_data, "amarsw-ck-B1.nc", 0);
  tot_flux += calc_flux_1band(ncol, nspecies, 150., 500., atm_data, "amarsw-ck-B2.nc", 1);
  tot_flux += calc_flux_1band(ncol, nspecies, 500., 1450., atm_data, "amarsw-ck-B3.nc", 0);
  tot_flux += calc_flux_1band(ncol, nspecies, 1450., 1850., atm_data, "amarsw-ck-B4.nc", 1);
  tot_flux += calc_flux_1band(ncol, nspecies, 1850., 3000., atm_data, "amarsw-ck-B5.nc", 0);
  std::cout << "tot_flux = " << tot_flux << std::endl;
}
