// C/C++
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// external
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/constants.h>

#include <harp/compound.hpp>
#include <harp/radiation/calc_dz_hypsometric.hpp>
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_band.hpp>
#include <harp/utils/find_resource.hpp>
#include <harp/utils/mean_molecular_weight.hpp>
#include <harp/utils/netcdf_opacity_utils.hpp>
#include <harp/utils/write_column_profile.hpp>
#include <harp/utils/write_spectral_profile.hpp>

namespace {

std::vector<double> tensor_to_vector(torch::Tensor tensor) {
  tensor = tensor.to(torch::kCPU).to(torch::kFloat64).contiguous();
  std::vector<double> out(tensor.numel());
  auto* ptr = tensor.data_ptr<double>();
  std::copy(ptr, ptr + tensor.numel(), out.begin());
  return out;
}

torch::Tensor load_band_wavenumber(harp::RadiationBandOptions const& options) {
  TORCH_CHECK(!options->opacities().empty(), "Band has no opacity sources");
  auto const& filename =
      options->opacities().begin()->second->opacity_files().front();

#ifdef NETCDFOUTPUT
  int fileid = harp::open_file(filename);
  int varid = -1;
  harp::check_nc(nc_inq_varid(fileid, "wavenumber", &varid),
                 "Missing required variable wavenumber");
  auto wavenumber = harp::convert_wavenumber_to_cm1(
      harp::read_1d_variable(fileid, "wavenumber"),
      harp::read_var_units(fileid, varid), "wavenumber");
  harp::check_nc(nc_close(fileid), "Failed to close NetCDF file");
  return wavenumber;
#else
  TORCH_CHECK(false, "NetCDF support is not enabled");
#endif
}

std::pair<std::vector<double>, std::vector<double>> make_bin_edges(
    std::vector<double> const& centers) {
  TORCH_CHECK(centers.size() >= 2, "Need at least two wavenumber points");

  std::vector<double> lower(centers.size()), upper(centers.size());
  for (size_t i = 0; i < centers.size(); ++i) {
    double left = 0.0;
    double right = 0.0;
    if (i == 0) {
      left = centers[0] - 0.5 * (centers[1] - centers[0]);
    } else {
      left = 0.5 * (centers[i - 1] + centers[i]);
    }
    if (i + 1 == centers.size()) {
      right = centers[i] + 0.5 * (centers[i] - centers[i - 1]);
    } else {
      right = 0.5 * (centers[i] + centers[i + 1]);
    }
    lower[i] = left;
    upper[i] = right;
  }
  return {lower, upper};
}

void configure_band_grid(harp::RadiationBandOptions const& options) {
  auto const wavenumber_tensor = load_band_wavenumber(options);
  auto const wavenumber = tensor_to_vector(wavenumber_tensor);
  auto const [wave_lower, wave_upper] = make_bin_edges(wavenumber);
  std::vector<double> weight(wavenumber.size());
  for (size_t i = 0; i < wavenumber.size(); ++i) {
    weight[i] = wave_upper[i] - wave_lower[i];
  }

  options->nwave(static_cast<int>(wavenumber.size()));
  options->wavenumber(wavenumber);
  options->weight(weight);
  options->set_wave_lower(wave_lower);
  options->set_wave_upper(wave_upper);
}

struct ColumnState {
  torch::Tensor pressure_levels;
  torch::Tensor temperature_levels;
  torch::Tensor pres;
  torch::Tensor temp;
  torch::Tensor conc;
  torch::Tensor dz;
};

ColumnState build_column(YAML::Node const& config) {
  TORCH_CHECK(config["problem"], "Missing 'problem' section");
  TORCH_CHECK(config["forcing"], "Missing 'forcing' section");
  TORCH_CHECK(config["forcing"]["const-gravity"],
              "Missing 'forcing.const-gravity' section");

  auto const problem = config["problem"];
  auto const gravity = config["forcing"]["const-gravity"];
  auto const nlyr = config["geometry"]["cells"]["nx1"].as<int>();

  harp::Composition composition_raw = {
      {"H2", problem["xH2"].as<double>()},
      {"He", problem["xHe"].as<double>()},
      {"CH4", problem["xCH4"].as<double>()},
      {"H2O", problem["xH2O"].as<double>()},
      {"NH3", problem["xNH3"].as<double>()},
  };
  auto const composition = harp::normalize_composition(composition_raw);

  auto const pbot = problem["Pbot"].as<double>();
  auto const ptop = problem["Ptop"].as<double>();
  auto const tbot = problem["Tbot"].as<double>();
  auto const ttop = problem["Ttop"].as<double>();
  auto const grav = std::abs(gravity["grav1"].as<double>());

  std::vector<double> pressure_levels(nlyr + 1);
  std::vector<double> temperature_levels(nlyr + 1);
  for (int i = 0; i <= nlyr; ++i) {
    double const frac = static_cast<double>(i) / static_cast<double>(nlyr);
    double const logp =
        std::log(pbot) + frac * (std::log(ptop) - std::log(pbot));
    pressure_levels[i] = std::exp(logp);
    temperature_levels[i] = tbot + frac * (ttop - tbot);
  }

  std::vector<double> pressure_layers(nlyr);
  std::vector<double> temperature_layers(nlyr);
  for (int i = 0; i < nlyr; ++i) {
    pressure_layers[i] = std::sqrt(pressure_levels[i] * pressure_levels[i + 1]);
    temperature_layers[i] =
        0.5 * (temperature_levels[i] + temperature_levels[i + 1]);
  }

  auto pressure_level_tensor = torch::tensor(pressure_levels, torch::kFloat64);
  auto pres = torch::tensor(pressure_layers, torch::kFloat64).unsqueeze(0);
  auto temperature_level_tensor =
      torch::tensor(temperature_levels, torch::kFloat64);
  auto temp = torch::tensor(temperature_layers, torch::kFloat64).unsqueeze(0);

  auto const total_molar_concentration = pres / (temp * harp::constants::Rgas);
  auto conc =
      torch::zeros({1, nlyr, static_cast<long>(harp::species_names.size())},
                   torch::kFloat64);
  for (size_t i = 0; i < harp::species_names.size(); ++i) {
    auto it = composition.find(harp::species_names[i]);
    double const mixing_ratio = (it == composition.end()) ? 0.0 : it->second;
    conc.index_put_({0, torch::indexing::Slice(), static_cast<long>(i)},
                    total_molar_concentration.squeeze(0) * mixing_ratio);
  }

  double mean_molecular_weight =
      harp::mean_molecular_weight(conc)[0].mean().item<double>();
  TORCH_CHECK(mean_molecular_weight > 0.0,
              "Mean molecular weight must be positive");

  auto g_over_r = torch::tensor(
      mean_molecular_weight * grav / harp::constants::Rgas, torch::kFloat64);
  auto dz = harp::calc_dz_hypsometric(pressure_level_tensor, temp.squeeze(0),
                                      g_over_r);

  return {
      pressure_level_tensor, temperature_level_tensor, pres, temp, conc, dz};
}

void print_layer_summary(std::string const& label,
                         torch::Tensor const& attenuation, int layer_index) {
  auto layer = attenuation.index({torch::indexing::Slice(), 0, layer_index, 0});
  std::cout << "  " << std::setw(12) << std::left << label
            << " mean[m^-1] = " << layer.mean().item<double>()
            << "  max[m^-1] = " << layer.max().item<double>() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path exe_dir = ".";
  if (argc > 0) {
    exe_dir = std::filesystem::absolute(argv[0]).parent_path();
    harp::add_resource_directory(exe_dir.string(), true);
  }

  auto yaml_path = (exe_dir / "jupiter_1d_rt.yaml").string();
  std::filesystem::path cli_output_dir = ".";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config") {
      TORCH_CHECK(i + 1 < argc, "--config requires a file path");
      yaml_path = argv[++i];
    } else if (arg == "--output-dir") {
      TORCH_CHECK(i + 1 < argc, "--output-dir requires a directory path");
      cli_output_dir = argv[++i];
    } else {
      TORCH_CHECK(false, "Unknown argument: ", arg,
                  ". Supported options are --config and --output-dir");
    }
  }

  auto yaml_dir = std::filesystem::absolute(yaml_path).parent_path();
  harp::add_resource_directory(yaml_dir.string(), true);

  auto config = YAML::LoadFile(yaml_path);
  auto rad_options = harp::RadiationOptionsImpl::from_yaml(yaml_path);
  TORCH_CHECK(rad_options->bands().size() == 1,
              "Example expects exactly one radiation band");

  auto band_options = rad_options->bands().front();
  configure_band_grid(band_options);
  harp::RadiationBand band(band_options);

  auto column = build_column(config);
  auto const reference_layer =
      config["problem"]["reference_layer"].as<int>(band_options->nlyr() / 2);
  TORCH_CHECK(reference_layer >= 0 && reference_layer < band_options->nlyr(),
              "Invalid reference_layer index: ", reference_layer);

  auto conc = column.conc;
  auto pres = column.pres;
  auto temp = column.temp;

  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = pres;
  atm["temp"] = temp;

  auto const ncol = band_options->ncol();
  auto const nwave = band_options->nwave();
  std::map<std::string, torch::Tensor> bc;
  bc[band_options->name() + "/albedo"] =
      torch::zeros({nwave, ncol}, torch::kFloat64);
  bc[band_options->name() + "/temis"] =
      torch::zeros({nwave, ncol}, torch::kFloat64);
  bc["btemp"] =
      torch::tensor({config["problem"]["surface_temperature_k"].as<double>(
                        config["problem"]["Tbot"].as<double>())},
                    torch::kFloat64);
  bc["ttemp"] =
      torch::tensor({config["problem"]["top_temperature_k"].as<double>(
                        config["problem"]["Ttop"].as<double>())},
                    torch::kFloat64);

  auto const wavenumber = band_options->wavenumber();
  std::vector<std::pair<std::string, torch::Tensor>> source_attenuation;
  std::vector<std::pair<std::string, torch::Tensor>> source_transmittance;
  torch::Tensor total_attenuation;
  bool first = true;
  for (auto& [name, module] : band->opacities) {
    auto attenuation = module.forward<torch::Tensor>(conc, atm);
    source_attenuation.push_back({name, attenuation});
    auto tau = (attenuation.squeeze(-1) * column.dz.view({1, 1, -1}))
                   .sum(-1)
                   .squeeze(1);
    source_transmittance.push_back({name, torch::exp(-tau)});
    if (first) {
      total_attenuation = attenuation.clone();
      first = false;
    } else {
      total_attenuation += attenuation;
    }
  }

  std::cout << "Band: " << band_options->name() << "\n";
  std::cout << "Layers = " << band_options->nlyr()
            << ", spectral points = " << band_options->nwave() << "\n";
  std::cout << "Reference layer = " << reference_layer << "\n";
  std::cout << "  pressure = " << pres[0][reference_layer].item<double>()
            << " Pa\n";
  std::cout << "  temperature = " << temp[0][reference_layer].item<double>()
            << " K\n";

  for (auto const& [name, attenuation] : source_attenuation) {
    print_layer_summary(name, attenuation, reference_layer);
  }
  print_layer_summary("total", total_attenuation, reference_layer);

  auto band_flux = band->forward(conc, column.dz.unsqueeze(0), &bc, &atm);
  auto upward = band_flux.index({0, torch::indexing::Slice(), 0});
  auto downward = band_flux.index({0, torch::indexing::Slice(), 1});
  auto net = upward - downward;

  std::cout << "\nIntegrated longwave flux profile [W/m^2]\n";
  std::cout << "  surface upward   = " << upward[0].item<double>() << "\n";
  std::cout << "  surface downward = " << downward[0].item<double>() << "\n";
  std::cout << "  surface net      = " << net[0].item<double>() << "\n";
  std::cout << "  TOA upward       = " << upward[-1].item<double>() << "\n";
  std::cout << "  TOA downward     = " << downward[-1].item<double>() << "\n";
  std::cout << "  TOA net          = " << net[-1].item<double>() << "\n";

  auto const output_dir =
      std::filesystem::absolute(cli_output_dir) /
      config["problem"]["output_dir"].as<std::string>("jupiter_1d_rt_outputs");
  harp::write_column_profile(output_dir / "column_profile.txt", column.dz, pres,
                             temp, conc, band_flux);
  harp::write_spectral_profile(
      output_dir / "spectral_profile.txt", wavenumber, source_transmittance,
      band->spectrum.index({torch::indexing::Slice(), 0, -1, 0}),
      band->spectrum.index({torch::indexing::Slice(), 0, 0, 1}));

  std::cout << "\nWrote diagnostics to " << output_dir.string() << "\n";
  return 0;
}
