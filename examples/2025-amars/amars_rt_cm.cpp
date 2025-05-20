// harp
#include <opacity/fourcolumn.hpp>
#include <opacity/rfm.hpp>
#include <radiation/calc_dz_hypsometric.hpp>
#include <radiation/disort_options_flux.hpp>
#include <radiation/flux_utils.hpp>
#include <rtsolver/rtsolver.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>
#include <utils/layer2level.hpp>
#include <utils/read_weights.hpp>

std::vector<std::vector<double>> read_4width_array_from_file(
    std::string fpath) {
  std::ifstream file(fpath);
  std::vector<std::vector<double>> array_to_get;
  std::string line;

  std::getline(file, line);  // skip the first line
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::vector<double> row;
    double value;
    while (iss >> value) {
      row.push_back(value);
    }
    if (row.size() == 4) {
      array_to_get.push_back(row);
    } else {
      std::cerr << "Invalid line format: " << line << std::endl;
    }
  }
  file.close();

  return array_to_get;
}

std::vector<double> read_values_from_file(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::vector<double> values;
  double value;
  while (file >> value) {
    values.push_back(value);
  }

  file.close();
  return values;
}

struct AtmosphericData {
  int n_layers;
  std::map<std::string, std::vector<double>> data;
};

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
      std::string name = line.substr(1);  // Remove the '*' character

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

torch::Tensor calc_dz_from_file(int nlyr, torch::Tensor prop,
                                AtmosphericData atm_data) {
  auto dz = torch::ones({nlyr, 1},
                        prop.options());  // prop: (nwave, ncol, nlyr, nprop)
  double z_i = 0;
  double z_i_plus_1 = 0;

  dz[0][0] = atm_data.data["HGT [km]"][0] * 1000;
  for (int i = 1; i < nlyr - 1; ++i) {
    z_i = atm_data.data["HGT [km]"][i];
    z_i_plus_1 = atm_data.data["HGT [km]"][i + 1];
    dz[i][0] *= (z_i_plus_1 - z_i) * 1000;
  }
  dz[nlyr - 1][0] *= (z_i_plus_1 - z_i) * 1000;
  return dz;
}

// unit = [mol/m^3]
torch::Tensor read_atm_concentration(int ncol, int nlyr, int nspecies,
                                     AtmosphericData atm_data) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  double pre = 0;
  double tem = 0;
  double R = 8.314472;
  for (int i = 0; i < nlyr; ++i) {
    pre = atm_data.data["PRE [mb]"][i] * 100.0;
    tem = atm_data.data["TEM [K]"][i];
    conc[0][i][0] = (atm_data.data["CO2 [ppmv]"][i] * 1e-6) * (pre / (R * tem));
    conc[0][i][1] = atm_data.data["H2O [ppmv]"][i] * 1e-6 * (pre / (R * tem));
    conc[0][i][2] = atm_data.data["SO2 [ppmv]"][i] * 1e-6 * (pre / (R * tem));
  }
  return conc;
}

torch::Tensor calc_flux_1band_init(int ncol, int nspecies, double wmin,
                                   double wmax, AtmosphericData atm_data,
                                   std::string filename, int idx,
                                   double btemp) {
  harp::AttenuatorOptions op;
  op.species_names({"CO2", "H2O", "SO2"});
  op.species_weights({44.0e-3, 18.0e-3, 64.0e-3});

  op.species_ids({0}).opacity_files({filename});
  harp::RFM co2(op);

  op.species_ids({1}).opacity_files({filename});
  harp::RFM h2o(op);

  op.species_ids({2}).opacity_files({filename});
  harp::RFM so2(op);

  int nwave = co2->kdata.size(0);
  int nlyr = atm_data.n_layers;

  auto conc = read_atm_concentration(ncol, nlyr, nspecies, atm_data);

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
  if (idx == 2) prop = so2->forward(conc, kwargs);
  auto dz = calc_dz_from_file(nlyr, prop, atm_data);

  prop *= dz;

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64) *
                 0.0;  // leave emissivity at 1
  bc["btemp"] = torch::ones({nwave, ncol}, torch::kFloat64) * btemp;

  auto temf = harp::layer2level(kwargs["temp"], harp::Layer2LevelOptions());
  auto flux = disort->forward(prop, &bc, temf);
  auto weights = harp::read_weights_rfm(filename);

  return (flux * weights.view({-1, 1, 1, 1})).sum(0);
}

torch::Tensor calc_flux_1band_loop(int ncol, int nspecies, double wmin,
                                   double wmax, AtmosphericData atm_data,
                                   std::string filename, int idx, double btemp,
                                   torch::Tensor new_T, torch::Tensor new_p,
                                   torch::Tensor dz) {
  harp::AttenuatorOptions op;
  op.species_names({"CO2", "H2O", "SO2"});
  op.species_weights({44.0e-3, 18.0e-3, 64.0e-3});

  op.species_ids({0}).opacity_files({filename});
  harp::RFM co2(op);

  op.species_ids({1}).opacity_files({filename});
  harp::RFM h2o(op);

  op.species_ids({2}).opacity_files({filename});
  harp::RFM so2(op);

  int nwave = co2->kdata.size(0);
  int nlyr = atm_data.n_layers;

  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  double pre = 0;
  double tem = 0;
  double R = 8.314472;
  for (int i = 0; i < nlyr; ++i) {
    pre = new_p[i];
    tem = new_T[i];
    conc[0][i][0] = (atm_data.data["CO2 [ppmv]"][i] * 1e-6) * (pre / (R * tem));
    conc[0][i][1] = atm_data.data["H2O [ppmv]"][i] * 1e-6 * (pre / (R * tem));
    conc[0][i][2] = atm_data.data["SO2 [ppmv]"][i] * 1e-6 * (pre / (R * tem));
  }

  disort::Disort disort(disort_options_lw(wmin, wmax, nwave, ncol, nlyr));

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  kwargs["pres"][0] = new_p;
  kwargs["temp"][0] = new_T;

  torch::Tensor prop;
  if (idx == 0) prop = co2->forward(conc, kwargs);
  if (idx == 1) prop = h2o->forward(conc, kwargs);
  if (idx == 2) prop = so2->forward(conc, kwargs);

  prop *= dz;

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64) *
                 0.0;  // leave emissivity at 1
  bc["btemp"] = torch::ones({nwave, ncol}, torch::kFloat64) * btemp;

  auto temf = harp::layer2level(kwargs["temp"], harp::Layer2LevelOptions());
  auto flux = disort->forward(prop, &bc, temf);
  auto weights = harp::read_weights_rfm(filename);

  return (flux * weights.view({-1, 1, 1, 1})).sum(0);
}

std::tuple<torch::Tensor, torch::Tensor> calc_flux_1band_loop_with_prop(
    int ncol, int nspecies, double wmin, double wmax, AtmosphericData atm_data,
    std::string filename, int idx, double btemp, std::vector<double> new_T,
    std::vector<double> new_p, torch::Tensor dz, int t_ind, int print_freq) {
  harp::AttenuatorOptions op;
  op.species_names({"CO2", "H2O", "SO2"});
  op.species_weights({44.0e-3, 18.0e-3, 64.0e-3});

  op.species_ids({0}).opacity_files({filename});
  harp::RFM co2(op);

  op.species_ids({1}).opacity_files({filename});
  harp::RFM h2o(op);

  op.species_ids({2}).opacity_files({filename});
  harp::RFM so2(op);

  int nwave = co2->kdata.size(0);
  int nlyr = atm_data.n_layers;

  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  double pre = 0;
  double tem = 0;
  double R = 8.314472;
  for (int i = 0; i < nlyr; ++i) {
    pre = new_p[i];
    tem = new_T[i];
    conc[0][i][0] = atm_data.data["CO2 [ppmv]"][i] * 1e-6 * (pre / (R * tem));
    conc[0][i][1] = atm_data.data["H2O [ppmv]"][i] * 1e-6 * (pre / (R * tem));
    conc[0][i][2] = atm_data.data["SO2 [ppmv]"][i] * 1e-6 * (pre / (R * tem));
  }

  disort::Disort disort(disort_options_lw(wmin, wmax, nwave, ncol, nlyr));

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64);
  for (int i = 0; i < nlyr; ++i) {
    kwargs["pres"][0][i] = new_p[i];
    kwargs["temp"][0][i] = new_T[i];
  }

  torch::Tensor prop;
  if (idx == 0) prop = co2->forward(conc, kwargs);
  if (idx == 1) prop = h2o->forward(conc, kwargs);
  if (idx == 2) prop = so2->forward(conc, kwargs);

  prop *= dz;

  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64) *
                 0.0;  // leave emissivity at 1
  bc["btemp"] = torch::ones({nwave, ncol}, torch::kFloat64) * btemp;

  auto temf = harp::layer2level(kwargs["temp"], harp::Layer2LevelOptions());
  auto flux = disort->forward(prop, &bc, temf);
  auto weights = harp::read_weights_rfm(filename);

  auto flux_result = (flux * weights.view({-1, 1, 1, 1})).sum(0);

  if (t_ind % print_freq == 0) {
    std::ostringstream filename3;
    filename3 << "conc_result_lw" << t_ind + 1 << ".txt";
    std::ofstream outputFile6(filename3.str());
    outputFile6 << "#p[Pa] conc_co2 conc_h2o conc_so2" << std::endl;
    for (int k = 0; k < nlyr; ++k) {
      outputFile6 << new_p[k] << " ";
      outputFile6 << conc[0][k][0].item<double>() << " ";
      outputFile6 << conc[0][k][1].item<double>() << " ";
      outputFile6 << conc[0][k][2].item<double>() << std::endl;
    }
    outputFile6 << std::endl;
    outputFile6.close();
  }

  return std::make_tuple(flux_result, prop);
}

double calculate_dynamic_timestep(const std::vector<double>& new_T,
                                  const std::vector<double>& dT_ds, int nlyr,
                                  double safety_factor) {
  double min_time_to_zero = std::numeric_limits<double>::max();
  bool found_negative = false;

  for (int k = 0; k < nlyr; ++k) {
    if (dT_ds[k] < 0) {
      double time_to_zero = new_T[k] / std::abs(dT_ds[k]);
      if (time_to_zero < min_time_to_zero) {
        min_time_to_zero = time_to_zero;
      }
      found_negative = true;
    }
  }

  if (!found_negative) {
    return 1e6;  // Return a large default timestep if no negative values are
                 // found
  }

  double timestep = min_time_to_zero / safety_factor;
  return (timestep > 1e6) ? 1e6 : timestep;
}

//// ------- shortwave ------- ////
void calc_shortwave(double wmin, double wmax, int nwave, int nlyr) {
  double solar_temp = 5772;
  double lum_scale = 0.7;
  int ncol = 1;
  int nwave = wave.size(0);

  // shortwave grid from 0.2um to 5um (2000 cm^-1 to 50000 cm^-1)
  auto wave = torch::linspace(2000, 50000, nwave, torch::kFloat64);

  RadiationBandOptions rt_sw_op;
  rt_sw_op.name() = "sw";
  rt_sw_op.solver_name() = "disort";
  rt_sw_op.opacity() = {
      {"h2so4",
       op.species_ids({0}).opacity_files({"h2so4.txt"}).type("h2so4_simple")},
      {"s8", op.species_ids({1})
                 .opacity_files({"s8_k_fuller.txt"})
                 .type("fourcolumn")}};
  rt_sw_op.disort() = disort(disort_flux_sw(nwave, ncol, nlyr));

  RadiationBand rt_sw(rt_sw_op);
  std::map<std::string, torch::Tensor> other;
  other["wavenumber"] = wave;

  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = lum_scale * sr_sun * bb_flux(wave, solar_temp, ncol);
  bc["umu0"] = 0.707 * torch::ones({nwave, ncol}, torch::kFloat64);
  bc["albedo"] = surf_sw_albedo * torch::ones({nwave, ncol}, torch::kFloat64);

  // shortwave flux at each wavenumber/level
  return rt_sw->forward(conc, dz, bc, other);
}

//// ------- longwave ------- ////
void calc_longwave_1band(std::string filename, double wmin, double wmax,
                         int nlyr) {
  RadiationBandOptions rt_lw_op;
  rt_lw_op.name() = "lw";
  rt_lw_op.solver_name() = "disort";
  rt_lw_op.opacity() = {
      {"CO2", op.species_ids({0}).opacity_files({filename}).type("rfm")},
      {"H2O", op.species_ids({1}).opacity_files({filename}).type("rfm")},
      {"SO2", op.species_ids({2}).opacity_files({filename}).type("rfm")}};
  rt_lw_op.disort() = disort_flux_lw(wmin, wmax, nwave, ncol, nlyr);

  RadiationBand rt_lw(rt_lw_op);

  // longwave flux at each wavenumber/level
  return rt_lw->forward(conc, dz, bc, other);
}

void calc_longwave(std::string filename, int nlyr) {
  auto tot_flux =
      calc_longwave_1band(1., 250., atm_data, "amars-ck-B1.nc", 0, btemp);
  tot_flux +=
      calc_flux_1band_init(250., 438., atm_data, "amars-ck-B2.nc", 1, btemp);
  tot_flux +=
      calc_flux_1band_init(438., 675., atm_data, "amars-ck-B3.nc", 2, btemp);
  tot_flux +=
      calc_flux_1band_init(675., 1062., atm_data, "amars-ck-B4.nc", 0, btemp);
  tot_flux +=
      calc_flux_1band_init(1062., 1200., atm_data, "amars-ck-B5.nc", 2, btemp);
  tot_flux +=
      calc_flux_1band_init(1200., 1600., atm_data, "amars-ck-B6.nc", 0, btemp);
  tot_flux +=
      calc_flux_1band_init(1600., 1900., atm_data, "amars-ck-B7.nc", 2, btemp);
  tot_flux +=
      calc_flux_1band_init(1900., 2000., atm_data, "amars-ck-B8.nc", 0, btemp);
}

int main(int argc, char** argv) {
  // int nwave = 48;
  int nwave = 500;  // 50 bins gets you within ~1 W/m^2 fldn at TOA, but we want
                    // to be sure to resolve spectral info
  // int nwave = 15000; //essentially the exact value of the integral over the
  // chosen wavelength bounds
  int ncol = 1;
  int nlyr = 40;  // 3 layers is 1 W/m^2 away from the exact value of fldn_surf
                  // when using 200 layers. however, we want some more layers to
                  // resolve heating
  int nspecies = 2;
  double g = 3.711;
  double mean_mol_weight = 0.044;  // CO2
  double R = 8.314472;
  double cp = 844;  // J/(kg K) for CO2
  double surf_sw_albedo = 0.3;
  // double aero_scale = 1e-6;
  double aero_scale = 1;
  double sr_sun = 2.92842e-5;  // angular size of the sun at mars

  //// ------- species ------- ////
  harp::AttenuatorOptions op;
  op.species_names({"H2SO4", "S8", "CO2", "H2O", "SO2"});
  op.species_weights({98.e-3, 256.e-3, 44.0e-3, 18.0e-3, 64.0e-3});

  // read in the atmos output, and extract pressure and mixing ratios
  // std::vector<std::vector<double>> aero_mr_p =
  //    read_4width_array_from_file("aerosol_output_data.txt");

  auto aero_mr_p = harp::read_data_tensor("aerosol_output_data.txt");
  aeros_mr_p.select(1, 0) *= 1e5;  // convert bar to Pa

  /*std::vector<double> p(aero_mr_p.size());
  std::vector<double> T(aero_mr_p.size());
  std::vector<std::vector<double>> mr(nspecies,
                                      std::vector<double>(aero_mr_p.size()));
  for (size_t i = 0; i < aero_mr_p.size(); ++i) {
    p[i] = aero_mr_p[i][0] * 1e5;  // convert bar to Pa
    T[i] = aero_mr_p[i][1];
    for (int j = 0; j < nspecies; ++j) {
      mr[j][i] = aero_mr_p[i][j + 2];
    }
  }*/

  // unit = [pa]
  auto new_P = harp::read_data_tensor("pVals.txt");
  new_P *= 100.0;  // convert mbar to Pa

  // unit = [K]
  auto new_T = harp::read_data_tensor("TVals.txt");

  // unit = [mol/mol]
  auto new_X = harp::interpn({new_P.log()}, {aero_mr_p.select(1, 0).log()},
                             aero_mr_p.narrow(1, 2, nspecies));

  // unit = [mol/m^3]
  auto conc = aero_scale * new_X * new_P / (R * new_T);

  // unit = [kg/m^3]
  auto new_rho = (new_P * mean_mol_weight) / (R * new_T);

  auto dz = calc_dz_hypsometric(new_p, new_T, torch::tensor({g / R}));

  auto total_flux_sw = harp::cal_total_flux(flux_sw, wave, "wave");
  auto surf_flux_sw = harp::cal_surface_flux(total_flux_sw);
  auto toa_flux_sw = harp::cal_toa_flux(total_flux_sw);

  // std::cout << "tot_flux_down_surf: " << tot_flux_down_surf << " W/m^2"
  //           << std::endl;
  // std::cout << "tot_flux_down_toa: " << tot_flux_down_toa << " W/m^2"
  //           << std::endl;

  //// ------- longwave ------- ////

  double btemp = 210;
  AtmosphericData atm_data = read_atmospheric_data("rfm.atm");
  int nlyr_lw = atm_data.n_layers;
  int nspecies_lw = 3;
  // you must pass the index of the species of the associated ck band
  // CO2 is idx 0, H2O is idx 1, SO2 idx is 2
  // std::cout << "tot_flux = " << tot_flux << std::endl;

  // calculate heating rates and write to file
  // unit = [K/s]
  auto dTdt = torch::zeros({nlyr}, torch::kFloat64);

  double df = 0;
  double df_iplus1 = 0;
  std::ofstream outputFile("dT_ds.txt");
  outputFile << "#p[Pa] dT_ds[K/s]" << std::endl;
  for (int k = 0; k < nlyr; ++k) {
    // add SW fluxes up/down to the LW fluxes up/down. 0 is up, 1 is down
    df = integrated_flux[k][0] - integrated_flux[k][1] +
         tot_flux[0][k][0].item<double>() - tot_flux[0][k][1].item<double>();
    df_iplus1 = integrated_flux[k + 1][0] - integrated_flux[k + 1][1] +
                tot_flux[0][k + 1][0].item<double>() -
                tot_flux[0][k + 1][1].item<double>();
    dT_ds[k] =
        -(1 / (new_rho[k] * cp)) * (df_iplus1 - df) / dz[k][0].item<double>();
    outputFile << new_p[k] << " " << dT_ds[k] << std::endl;
  }
  outputFile.close();

  std::ostringstream filename;
  filename << "tp_result" << 0 << ".txt";

  // Open the file and write the data
  std::ofstream outputFile_init(filename.str());
  outputFile_init << "#p[Pa] T[K] dT/dt [K/s]" << std::endl;
  for (int k = 0; k < nlyr; ++k) {
    outputFile_init << new_p[k] << " " << new_T[k] << " " << dT_ds[k]
                    << std::endl;
  }
  outputFile_init.close();

  // CALC THE RAD EQUIL
  int t_lim = 50000;  // 10000 = 6 yr with 6hr timesteps
  int print_freq = 500;
  double tstep = 86400 / 1;  // step is 6hrs
  double cSurf =
      200000;  // thermal inertia of the surface, assuming half ocean half land
  double current_time = 0;
  std::ofstream outputFile2("btemp_t.txt");
  outputFile2 << "#t[s] btemp[K]" << std::endl;
  outputFile2 << current_time << " " << btemp << std::endl;

  // start the time loop
  for (int t_ind = 0; t_ind < t_lim; ++t_ind) {
    // double tstep = calculate_dynamic_timestep(new_T, dT_ds, nlyr, 1000);
    // std::cout << "tstep = " << tstep << std::endl;
    // double tstep = 86400;

    current_time += tstep;
    // sw + lw flux down - surf emission
    double surf_forcing = (1 - surf_sw_albedo) * integrated_flux[0][1] +
                          tot_flux[0][0][1].item<double>() -
                          5.67e-8 * std::pow(btemp, 4);
    btemp += surf_forcing * (tstep / cSurf);

    for (int k = 0; k < nlyr; ++k) {
      new_T[k] += dT_ds[k] * tstep;
      if (new_T[k] < 20) new_T[k] = 20;

      // PRESSURE GRID IS FIXED
      conc[0][k][0] =
          (aero_scale * new_mr[1][k] * new_p[k]) /
          (R * new_T[k]);  // s8 comes second in the file that we read
                           // in. but we need it to be index 0 in conc
                           // bc of how it was initialized above
      conc[0][k][1] =
          (aero_scale * new_mr[0][k] * new_p[k]) /
          (R * new_T[k]);  // h2so4 comes first in the file that we read in. but
                           // it needs to be index 1 in conc.
      new_rho[k] = (new_p[k] * mean_mol_weight) / (R * new_T[k]);
    }

    auto prop1 = s8->forward(conc, kwargs);
    auto prop2 = h2so4->forward(conc, kwargs);
    auto prop = prop1 + prop2;

    auto dz = calc_dz(nlyr, prop, new_p, new_rho, g);
    prop *= dz;

    // mean single scattering albedo
    prop.select(3, 1) /= prop.select(3, 0);

    auto result = disort->forward(prop, &bc);

    auto [integrated_flux, tot_flux_down_surf, tot_flux_down_toa] =
        integrate_result(result, wave, nlyr, nwave);

    if (t_ind % print_freq == 0) {
      std::ostringstream filename3;
      filename3 << "conc_result_sw" << t_ind + 1 << ".txt";
      std::ofstream outputFile6(filename3.str());
      outputFile6 << "#p[Pa] conc_s8 conc_h2so4" << std::endl;
      for (int k = 0; k < nlyr; ++k) {
        outputFile6 << new_p[k] << " ";
        outputFile6 << conc[0][k][0].item<double>() << " ";
        outputFile6 << conc[0][k][1].item<double>() << " ";
      }
      outputFile6 << std::endl;
      outputFile6.close();
    }

    std::vector<torch::Tensor> prop_results;
    torch::Tensor tot_flux;
    torch::Tensor flux_result, prop_result;

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 1., 250., atm_data, "amars-ck-B1.nc", 0, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux = flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 250., 438., atm_data, "amars-ck-B2.nc", 1, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 438., 675., atm_data, "amars-ck-B3.nc", 2, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 675., 1062., atm_data, "amars-ck-B4.nc", 0, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 1062., 1200., atm_data, "amars-ck-B5.nc", 2, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 1200., 1600., atm_data, "amars-ck-B6.nc", 0, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 1600., 1900., atm_data, "amars-ck-B7.nc", 2, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    std::tie(flux_result, prop_result) = calc_flux_1band_loop_with_prop(
        ncol, nspecies_lw, 1900., 2000., atm_data, "amars-ck-B8.nc", 0, btemp,
        new_T, new_p, dz, t_ind, print_freq);
    tot_flux += flux_result;
    prop_results.push_back(prop_result);

    // calculate heating rates
    double df = 0;
    double df_iplus1 = 0;
    for (int k = 0; k < nlyr; ++k) {
      // add SW fluxes up/down to the LW fluxes up/down. 0 is up, 1 is down
      df = integrated_flux[k][0] - integrated_flux[k][1] +
           tot_flux[0][k][0].item<double>() - tot_flux[0][k][1].item<double>();
      df_iplus1 = integrated_flux[k + 1][0] - integrated_flux[k + 1][1] +
                  tot_flux[0][k + 1][0].item<double>() -
                  tot_flux[0][k + 1][1].item<double>();
      dT_ds[k] =
          -(1 / (new_rho[k] * cp)) * (df_iplus1 - df) / dz[k][0].item<double>();
    }
    outputFile2 << current_time << " " << btemp << std::endl;

    if (t_ind % print_freq == 0) {
      std::ostringstream filename;
      filename << "tp_result" << t_ind + 1 << ".txt";

      // Open the file and write the data
      std::ofstream outputFile3(filename.str());
      outputFile3 << "#p[Pa] T[K] dT/dt [K/s]" << std::endl;
      for (int k = 0; k < nlyr; ++k) {
        outputFile3 << new_p[k] << " " << new_T[k] << " " << dT_ds[k]
                    << std::endl;
      }
      outputFile3.close();

      std::ostringstream filename2;
      filename2 << "tau_result" << t_ind + 1 << ".txt";
      std::ofstream outputFile5(filename2.str());
      outputFile5 << "#p[Pa] tau_sw ssa_sw tau_b1 tau_b2 tau_b3 tau_b4 tau_b5 "
                     "tau_b6 tau_b7 tau_b8"
                  << std::endl;
      for (int k = 0; k < nlyr; ++k) {
        outputFile5 << new_p[k] << " ";
        outputFile5 << prop.select(3, 0)[0][0][k].item<double>() << " ";
        outputFile5 << prop.select(3, 1)[0][0][k].item<double>() << " ";
        for (int i = 0; i < 8; ++i) {
          outputFile5 << prop_results[i].select(3, 0)[0][0][k].item<double>()
                      << " ";
        }
        outputFile5 << std::endl;
      }
      outputFile5.close();
    }
  }
  outputFile2.close();

  std::ofstream outputFile4("final_tp_result.txt");
  outputFile4 << "#p[Pa] T[K]" << std::endl;
  for (int k = 0; k < nlyr; ++k) {
    outputFile4 << new_p[k] << " " << new_T[k] << std::endl;
  }
  outputFile4.close();
}
