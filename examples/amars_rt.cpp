#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// harp
#include <add_arg.h>
#include <configure.h>
#include <math/interpn.h>

#include <math/interpolation.hpp>
#include <opacity/h2so4_simple.hpp>
#include <opacity/s8_fuller.hpp>
#include <rtsolver/rtsolver.hpp>
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

namespace harp {

void interpolate_mixing_ratios(const std::vector<double>& dense_grid,
                               const std::vector<double>& mixing_ratios,
                               const std::vector<double>& thinned_grid,
                               std::vector<double>& interpolated_ratios) {
  // Ensure the output vector is the correct size
  interpolated_ratios.resize(thinned_grid.size());

  // Perform 1D interpolation for each point in the thinned grid
  for (size_t i = 0; i < thinned_grid.size(); ++i) {
    interpolated_ratios[i] = interp1(thinned_grid[i], mixing_ratios.data(),
                                     dense_grid.data(), dense_grid.size());
  }
}

}  // namespace harp

// disort
#include <disort/disort.hpp>

disort::DisortOptions disort_options(int nwave, int ncol, int nlyr) {
  disort::DisortOptions op;

  op.header("running amars RT");
  op.flags(
      "lamber,quiet,onlyfl,"
      "intensity_correction,old_intensity_correction");
  //"intensity_correction,old_intensity_correction,"
  //"print-input,print-phase-function");

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

// unit = [mol/m^3]
torch::Tensor atm_concentration(int ncol, int nlyr, int nspecies) {
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  return conc;
}

// unit = [cm^-1]
torch::Tensor short_wavenumber_grid(int nwave) {
  int wmin = 2000;   // 5 um
  int wmax = 50000;  // 200 nm this gets us within 2 W/m^2 of the correct 410
                     // value. this is where rayleigh attenuates all intensity
  return torch::linspace(wmin, wmax, nwave, torch::kFloat64);
}

// unit = [w/(m^2 cm^-1)]
torch::Tensor short_toa_flux(int nwave, int ncol) {
  return torch::ones({nwave, ncol}, torch::kFloat64);
}

// unit = [w/(m^2 cm^-1)]
torch::Tensor bb_toa_flux(torch::Tensor wave, int ncol, double temp,
                          double fscale) {
  double c1 = 1.19144e-5 * 1e-3;
  double c2 = 1.4388;
  double sr_sun = 2.92842e-5;  // angular size of the sun at mars

  int nwave = wave.size(0);
  torch::Tensor bb_flux = torch::ones({nwave, ncol}, torch::kFloat64);
  for (int i = 0; i < nwave; ++i) {
    for (int j = 0; j < ncol; ++j) {
      bb_flux[i][j] = fscale * sr_sun * c1 * pow(wave[i], 3) /
                      (exp(c2 * wave[i] / temp) - 1);
    }
  }
  return bb_flux;
}

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

std::tuple<std::vector<double>, std::vector<double>,
           std::vector<std::vector<double>>>
regrid_ptx(int nlyr, int nspecies, std::vector<double> p, std::vector<double> T,
           std::vector<std::vector<double>> mr) {
  double p_min = *std::min_element(p.begin(), p.end());
  double p_max = *std::max_element(p.begin(), p.end());
  double p_step = (p_max - p_min) / (nlyr - 1);

  double T_min = *std::min_element(T.begin(), T.end());
  double T_max = *std::max_element(T.begin(), T.end());
  double T_step = (T_max - T_min) / (nlyr - 1);

  std::vector<double> new_p(nlyr);
  std::vector<double> new_T(nlyr);
  for (int i = 0; i < nlyr; ++i) {
    new_p[nlyr - 1 - i] = p_min + i * p_step;
    new_T[nlyr - 1 - i] = T_min + i * T_step;
  }

  std::vector<std::vector<double>> new_mr(nspecies, std::vector<double>(nlyr));
  for (int j = 0; j < nspecies; ++j) {
    harp::interpolate_mixing_ratios(p, mr[j], new_p, new_mr[j]);
  }

  return std::make_tuple(new_p, new_T, new_mr);
}

torch::Tensor calc_dz(int nlyr, torch::Tensor prop,
                      const std::vector<double>& new_p,
                      const std::vector<double>& new_rho, double g) {
  auto dz = torch::ones({nlyr, 1},
                        prop.options());  // prop: (nwave, ncol, nlyr, nprop)
  for (int i = 0; i < nlyr - 1; ++i) {
    double dp = new_p[i] - new_p[i + 1];
    dz[i][0] *= (dp) / (g * new_rho[i]);
  }
  // ignoring the p grid/z grid last layer issue. really p is in center and z is
  // on the border
  // we barely have any attenuration from the last layer anyway so just ignore
  // this issue
  dz[nlyr - 1][0] *= 2 * dz[nlyr - 2][0];

  return dz;
}

std::tuple<std::vector<std::vector<double>>, double, double> integrate_result(
    const torch::Tensor& result, const torch::Tensor& wave, int nlyr,
    int nwave) {
  std::vector<std::vector<double>> integrated_flux(nlyr + 1,
                                                   std::vector<double>(2, 0.0));
  // assume a constant wavenumber grid
  double d_lambda = wave[1].item<double>() - wave[0].item<double>();
  double tot_flux_down_surf = 0;
  double tot_flux_down_toa = 0;
  for (int i = 0; i < nwave; ++i) {
    for (int k = 0; k <= nlyr; ++k) {
      integrated_flux[k][0] += result[i][0][k][0].item<double>() *
                               d_lambda;  // result(x,x,x,0) is flux_up
      integrated_flux[k][1] +=
          result[i][0][k][1].item<double>() * d_lambda;  // 1 is down
    }
    tot_flux_down_surf += result[i][0][0][1].item<double>() * d_lambda;
    tot_flux_down_toa += result[i][0][nlyr][1].item<double>() * d_lambda;
  }

  return std::make_tuple(integrated_flux, tot_flux_down_surf,
                         tot_flux_down_toa);
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
  double solar_temp = 5772;
  double lum_scale = 0.7;

  disort::Disort disort(disort_options(nwave, ncol, nlyr));

  harp::AttenuatorOptions op;
  op.species_names({"S8", "H2SO4"});
  op.species_weights({256.e-3, 98.e-3});

  op.species_ids({0}).opacity_files({"s8_k_fuller.txt"});
  harp::S8Fuller s8(op);

  op.species_ids({1}).opacity_files({"h2so4.txt"});
  harp::H2SO4Simple h2so4(op);

  auto wave = short_wavenumber_grid(nwave);
  auto conc = atm_concentration(ncol, nlyr, nspecies);

  // read in the atmos output, and extract pressure and mixing ratios
  std::vector<std::vector<double>> aero_mr_p =
      read_4width_array_from_file("aerosol_output_data.txt");
  std::vector<double> p(aero_mr_p.size());
  std::vector<double> T(aero_mr_p.size());
  std::vector<std::vector<double>> mr(nspecies,
                                      std::vector<double>(aero_mr_p.size()));
  for (size_t i = 0; i < aero_mr_p.size(); ++i) {
    p[i] = aero_mr_p[i][0] * 1e5;  // convert bar to Pa
    T[i] = aero_mr_p[i][1];
    for (int j = 0; j < nspecies; ++j) {
      mr[j][i] = aero_mr_p[i][j + 2];
    }
  }

  auto [new_p, new_T, new_mr] = regrid_ptx(nlyr, nspecies, p, T, mr);
  std::vector<double> new_rho(nlyr);
  std::vector<double> dT_ds(nlyr);

  for (int k = 0; k < nlyr; ++k) {
    conc[0][k][0] = (new_mr[1][k] * new_p[k]) /
                    (R * new_T[k]);  // s8 comes second in the file that we read
                                     // in. but we need it to be index 0 in conc
                                     // bc of how it was initialized above
    conc[0][k][1] =
        (new_mr[0][k] * new_p[k]) /
        (R * new_T[k]);  // h2so4 comes first in the file that we read in. but
                         // it needs to be index 1 in conc.
    new_rho[k] = (new_p[k] * mean_mol_weight) / (R * new_T[k]);
  }

  std::map<std::string, torch::Tensor> kwargs;
  kwargs["wavenumber"] = wave;

  auto prop1 = s8->forward(conc, kwargs);
  auto prop2 = h2so4->forward(conc, kwargs);
  auto prop = prop1 + prop2;

  auto dz = calc_dz(nlyr, prop, new_p, new_rho, g);
  prop *= dz;

  // mean single scattering albedo
  prop.select(3, 1) /= prop.select(3, 0);

  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = bb_toa_flux(wave, ncol, solar_temp, lum_scale);
  bc["umu0"] = 1.0 * torch::ones({nwave, ncol}, torch::kFloat64);
  bc["albedo"] = 1.0 * torch::ones({nwave, ncol}, torch::kFloat64);

  auto result = disort->forward(prop, &bc);

  auto [integrated_flux, tot_flux_down_surf, tot_flux_down_toa] =
      integrate_result(result, wave, nlyr, nwave);

  std::cout << "tot_flux_down_surf: " << tot_flux_down_surf << " W/m^2"
            << std::endl;
  std::cout << "tot_flux_down_toa: " << tot_flux_down_toa << " W/m^2"
            << std::endl;

  // calculate heating rates and write to file
  double df = 0;
  double df_iplus1 = 0;
  std::ofstream outputFile("dT_ds.txt");
  outputFile << "#p[Pa] dT_ds[K/s]" << std::endl;
  for (int k = 0; k < nlyr; ++k) {
    df = integrated_flux[k][0] - integrated_flux[k][1];
    df_iplus1 = integrated_flux[k + 1][0] - integrated_flux[k + 1][1];
    dT_ds[k] =
        -(1 / (new_rho[k] * cp)) * (df_iplus1 - df) / dz[k][0].item<double>();
    outputFile << new_p[k] << " " << dT_ds[k] << std::endl;
  }
  outputFile.close();

  // make sure the mixing ratio interpolation is working fine
  std::ofstream outputFile2("mix.txt");
  outputFile2 << "#p_new[Pa] mr(new)(0) mr(new)(1) p_atmos[Pa] mr(atmos)(0) "
                 "mr(atmos)(1)"
              << std::endl;
  for (int k = 0; k < nlyr; ++k) {
    outputFile2 << new_p[k] << " " << new_mr[0][k] << " " << new_mr[1][k] << " "
                << p[k] << " " << mr[0][k] << " " << mr[1][k] << std::endl;
  }
  for (int k = nlyr; k < 200; ++k) {
    outputFile2 << 0 << " " << 0 << " " << 0 << " " << p[k] << " " << mr[0][k]
                << " " << mr[1][k] << std::endl;
  }
  outputFile2.close();
}
