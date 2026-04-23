#pragma once

// C/C++
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

namespace harp {

inline void write_spectral_profile(
    std::filesystem::path const& filename,
    std::vector<double> const& wavenumber,
    std::vector<std::pair<std::string, torch::Tensor>> const& transmittance,
    torch::Tensor const& flux_up_toa, torch::Tensor const& flux_down_boa) {
  auto fup = flux_up_toa.to(torch::kCPU).to(torch::kFloat64).contiguous();
  auto fdn = flux_down_boa.to(torch::kCPU).to(torch::kFloat64).contiguous();

  TORCH_CHECK(fup.dim() == 1, "flux_up_toa must be 1D");
  TORCH_CHECK(fdn.dim() == 1, "flux_down_boa must be 1D");
  TORCH_CHECK(static_cast<int64_t>(wavenumber.size()) == fup.size(0),
              "wavenumber size mismatch with flux_up_toa");
  TORCH_CHECK(fup.size(0) == fdn.size(0),
              "flux_up_toa and flux_down_boa size mismatch");

  std::filesystem::create_directories(filename.parent_path());
  std::ofstream out(filename);
  TORCH_CHECK(out, "Failed to open output file: ", filename.string());
  int const value_width = 16;

  out << "# WNO: wavenumber [cm^-1]\n";
  for (auto const& [name, values] : transmittance) {
    out << "# " << name << ": total transmittance from " << name << " [-]\n";
  }
  out << "# FUP: upward spectral flux at TOA [W/m^2/cm^-1]\n";
  out << "# FDN: downward spectral flux at BOA [W/m^2/cm^-1]\n";

  out << std::setw(value_width) << "WNO";
  std::vector<torch::Tensor> trans_cpu;
  for (auto const& [name, values] : transmittance) {
    auto tensor = values.to(torch::kCPU).to(torch::kFloat64).contiguous();
    TORCH_CHECK(tensor.dim() == 1, "transmittance tensor for ", name,
                " must be 1D");
    TORCH_CHECK(tensor.size(0) == static_cast<int64_t>(wavenumber.size()),
                "transmittance size mismatch for ", name);
    trans_cpu.push_back(tensor);
    out << std::setw(
               std::max<int>(value_width, static_cast<int>(name.size()) + 2))
        << name;
  }
  out << std::setw(value_width) << "FUP" << std::setw(value_width) << "FDN"
      << "\n";

  out << std::setprecision(8);
  for (size_t i = 0; i < wavenumber.size(); ++i) {
    out << std::setw(value_width) << std::defaultfloat << wavenumber[i];
    for (size_t j = 0; j < trans_cpu.size(); ++j) {
      auto const& name = transmittance[j].first;
      out << std::setw(
                 std::max<int>(value_width, static_cast<int>(name.size()) + 2))
          << trans_cpu[j][static_cast<long>(i)].item<double>();
    }
    out << std::setw(value_width) << fup[static_cast<long>(i)].item<double>();
    out << std::setw(value_width) << fdn[static_cast<long>(i)].item<double>()
        << "\n";
  }
}

}  // namespace harp
