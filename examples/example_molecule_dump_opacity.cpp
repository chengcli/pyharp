// C/C++
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// external
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// harp
#include <harp/constants.h>

#include <harp/radiation/calc_dz_hypsometric.hpp>
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_band.hpp>
#include <harp/utils/find_resource.hpp>
#include <harp/utils/netcdf_opacity_utils.hpp>

namespace {

using Composition = std::map<std::string, double>;

Composition parse_composition_string(std::string const& text) {
  Composition composition;
  std::stringstream ss(text);
  std::string token;

  while (std::getline(ss, token, ',')) {
    auto const colon = token.find(':');
    TORCH_CHECK(colon != std::string::npos,
                "Invalid composition entry: ", token);
    composition[token.substr(0, colon)] = std::stod(token.substr(colon + 1));
  }
  return composition;
}

double sum_composition(Composition const& composition) {
  double total = 0.0;
  for (auto const& [name, value] : composition) total += value;
  return total;
}

Composition normalize_composition(Composition const& composition) {
  auto const total = sum_composition(composition);
  TORCH_CHECK(total > 0.0, "Composition sum must be positive");

  Composition normalized;
  for (auto const& [name, value] : composition)
    normalized[name] = value / total;
  return normalized;
}

std::string format_composition(Composition const& composition) {
  std::ostringstream oss;
  bool first = true;
  for (auto const& [name, value] : composition) {
    if (!first) oss << ",";
    first = false;
    oss << name << ":" << value;
  }
  return oss.str();
}

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
  torch::Tensor pressure_layers;
  torch::Tensor temperature_levels;
  torch::Tensor temperature_layers;
  torch::Tensor concentrations;
  torch::Tensor dz;
  double mean_molecular_weight_kg_mol;
  Composition composition;
  double raw_composition_sum;
};

ColumnState build_column(YAML::Node const& config) {
  auto const state = config["state"];
  auto const nlyr = config["geometry"]["cells"]["nx1"].as<int>();
  auto const composition_raw =
      parse_composition_string(state["composition"].as<std::string>());
  auto const raw_sum = sum_composition(composition_raw);
  auto const composition = normalize_composition(composition_raw);

  auto const pbot = state["pressure_bottom_pa"].as<double>();
  auto const ptop = state["pressure_top_pa"].as<double>();
  auto const tbot = state["temperature_bottom_k"].as<double>();
  auto const ttop = state["temperature_top_k"].as<double>();
  auto const grav = state["gravity_m_s2"].as<double>();

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

  auto pressure_level_tensor =
      torch::tensor(pressure_levels, torch::kFloat64).unsqueeze(0);
  auto pressure_layer_tensor =
      torch::tensor(pressure_layers, torch::kFloat64).unsqueeze(0);
  auto temperature_level_tensor =
      torch::tensor(temperature_levels, torch::kFloat64).unsqueeze(0);
  auto temperature_layer_tensor =
      torch::tensor(temperature_layers, torch::kFloat64).unsqueeze(0);

  double mean_molecular_weight = 0.0;
  for (size_t i = 0; i < harp::species_names.size(); ++i) {
    auto it = composition.find(harp::species_names[i]);
    if (it != composition.end()) {
      mean_molecular_weight += it->second * harp::species_weights[i];
    }
  }
  TORCH_CHECK(mean_molecular_weight > 0.0,
              "Mean molecular weight must be positive");

  auto const total_molar_concentration =
      pressure_layer_tensor /
      (temperature_layer_tensor * harp::constants::Rgas);
  auto concentrations =
      torch::zeros({1, nlyr, static_cast<long>(harp::species_names.size())},
                   torch::kFloat64);
  for (size_t i = 0; i < harp::species_names.size(); ++i) {
    auto it = composition.find(harp::species_names[i]);
    double const mixing_ratio = (it == composition.end()) ? 0.0 : it->second;
    concentrations.index_put_(
        {0, torch::indexing::Slice(), static_cast<long>(i)},
        total_molar_concentration.squeeze(0) * mixing_ratio);
  }

  auto g_over_r = torch::tensor(
      mean_molecular_weight * grav / harp::constants::Rgas, torch::kFloat64);
  auto dz = harp::calc_dz_hypsometric(pressure_level_tensor,
                                      temperature_layer_tensor, g_over_r);

  return {
      pressure_level_tensor,    pressure_layer_tensor, temperature_level_tensor,
      temperature_layer_tensor, concentrations,        dz,
      mean_molecular_weight,    composition,           raw_sum};
}

void print_layer_summary(std::string const& label,
                         torch::Tensor const& attenuation, int layer_index) {
  auto layer = attenuation.index({torch::indexing::Slice(), 0, layer_index, 0});
  std::cout << "  " << std::setw(12) << std::left << label
            << " mean[m^-1] = " << layer.mean().item<double>()
            << "  max[m^-1] = " << layer.max().item<double>() << "\n";
}

void write_reference_layer_csv(
    std::filesystem::path const& output_dir, int layer_index,
    std::vector<double> const& wavenumber,
    std::vector<std::pair<std::string, torch::Tensor>> const& spectra,
    torch::Tensor const& total) {
  std::filesystem::create_directories(output_dir);
  auto path = output_dir / ("reference_layer_" + std::to_string(layer_index) +
                            "_attenuation.csv");
  std::ofstream out(path);
  TORCH_CHECK(out, "Failed to open output file: ", path.string());

  out << "wavenumber_cm-1";
  for (auto const& [name, _] : spectra) out << "," << name << "_m-1";
  out << ",total_m-1\n";

  out << std::setprecision(12);
  auto total_layer =
      total.index({torch::indexing::Slice(), 0, layer_index, 0}).contiguous();
  std::vector<torch::Tensor> source_layers;
  for (auto const& [name, tensor] : spectra) {
    source_layers.push_back(
        tensor.index({torch::indexing::Slice(), 0, layer_index, 0})
            .contiguous());
  }

  for (size_t i = 0; i < wavenumber.size(); ++i) {
    out << wavenumber[i];
    for (auto const& source : source_layers)
      out << "," << source[i].item<double>();
    out << "," << total_layer[static_cast<long>(i)].item<double>() << "\n";
  }
}

void write_flux_profile_csv(std::filesystem::path const& output_dir,
                            torch::Tensor const& pressure_levels,
                            torch::Tensor const& band_flux) {
  std::filesystem::create_directories(output_dir);
  auto path = output_dir / "flux_profile.csv";
  std::ofstream out(path);
  TORCH_CHECK(out, "Failed to open output file: ", path.string());

  auto level_pressure = pressure_levels.squeeze(0).contiguous();
  auto upward = band_flux.index({0, torch::indexing::Slice(), 0}).contiguous();
  auto downward =
      band_flux.index({0, torch::indexing::Slice(), 1}).contiguous();
  auto net = upward - downward;

  out << "level,pressure_pa,upward_flux_w_m-2,downward_flux_w_m-2,net_flux_w_m-"
         "2\n";
  out << std::setprecision(12);
  for (int64_t i = 0; i < level_pressure.size(0); ++i) {
    out << i << "," << level_pressure[i].item<double>() << ","
        << upward[i].item<double>() << "," << downward[i].item<double>() << ","
        << net[i].item<double>() << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path exe_dir = ".";
  if (argc > 0) {
    exe_dir = std::filesystem::absolute(argv[0]).parent_path();
    harp::add_resource_directory(exe_dir.string(), true);
  }

  auto yaml_path = (exe_dir / "molecule_dump_opacity.yaml").string();
  if (argc > 1) yaml_path = argv[1];

  auto config = YAML::LoadFile(yaml_path);
  auto rad_options = harp::RadiationOptionsImpl::from_yaml(yaml_path);
  TORCH_CHECK(rad_options->bands().size() == 1,
              "Example expects exactly one radiation band");

  auto band_options = rad_options->bands().front();
  configure_band_grid(band_options);
  harp::RadiationBand band(band_options);

  auto column = build_column(config);
  auto const reference_layer =
      config["state"]["reference_layer"].as<int>(band_options->nlyr() / 2);
  TORCH_CHECK(reference_layer >= 0 && reference_layer < band_options->nlyr(),
              "Invalid reference_layer index: ", reference_layer);

  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = column.pressure_layers;
  atm["temp"] = column.temperature_layers;

  auto const ncol = band_options->ncol();
  auto const nwave = band_options->nwave();
  std::map<std::string, torch::Tensor> bc;
  bc[band_options->name() + "/albedo"] =
      torch::zeros({nwave, ncol}, torch::kFloat64);
  bc[band_options->name() + "/temis"] =
      torch::zeros({nwave, ncol}, torch::kFloat64);
  bc["btemp"] = torch::tensor(
      {config["state"]["surface_temperature_k"].as<double>()}, torch::kFloat64);
  bc["ttemp"] = torch::tensor(
      {config["state"]["top_temperature_k"].as<double>()}, torch::kFloat64);

  auto const wavenumber = band_options->wavenumber();
  std::vector<std::pair<std::string, torch::Tensor>> source_attenuation;
  torch::Tensor total_attenuation;
  bool first = true;
  for (auto& [name, module] : band->opacities) {
    auto attenuation =
        module.forward<torch::Tensor>(column.concentrations, atm);
    source_attenuation.push_back({name, attenuation});
    if (first) {
      total_attenuation = attenuation.clone();
      first = false;
    } else {
      total_attenuation += attenuation;
    }
  }

  std::cout << "Band: " << band_options->name() << "\n";
  std::cout << "Species composition = "
            << config["state"]["composition"].as<std::string>() << "\n";
  std::cout << "Composition sum = " << column.raw_composition_sum << "\n";
  if (std::abs(column.raw_composition_sum - 1.0) > 1.e-12) {
    std::cout << "Normalized composition = "
              << format_composition(column.composition) << "\n";
  }
  std::cout << "Mean molecular weight = " << column.mean_molecular_weight_kg_mol
            << " kg/mol\n";
  std::cout << "Layers = " << band_options->nlyr()
            << ", spectral points = " << band_options->nwave() << "\n";
  std::cout << "Reference layer = " << reference_layer << "\n";
  std::cout << "  pressure = "
            << column.pressure_layers[0][reference_layer].item<double>()
            << " Pa\n";
  std::cout << "  temperature = "
            << column.temperature_layers[0][reference_layer].item<double>()
            << " K\n";

  for (auto const& [name, attenuation] : source_attenuation) {
    print_layer_summary(name, attenuation, reference_layer);
  }
  print_layer_summary("total", total_attenuation, reference_layer);

  auto band_flux = band->forward(column.concentrations, column.dz, &bc, &atm);
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
      exe_dir / config["state"]["output_dir"].as<std::string>(
                    "molecule_dump_band_outputs");
  write_reference_layer_csv(output_dir, reference_layer, wavenumber,
                            source_attenuation, total_attenuation);
  write_flux_profile_csv(output_dir, column.pressure_levels, band_flux);

  std::cout << "\nWrote diagnostics to " << output_dir.string() << "\n";
  return 0;
}
