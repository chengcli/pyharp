#pragma once

// C/C++
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// harp
#include <harp/utils/mean_molecular_weight.hpp>

namespace harp {

extern std::vector<std::string> species_names;

inline void write_column_profile(std::filesystem::path const& filename,
                                 torch::Tensor const& dz,
                                 torch::Tensor const& pres,
                                 torch::Tensor const& temp,
                                 torch::Tensor const& conc,
                                 torch::Tensor const& band_flux) {
  auto dz_1d = dz.to(torch::kCPU).to(torch::kFloat64).contiguous();
  auto pres_2d = pres.to(torch::kCPU).to(torch::kFloat64).contiguous();
  auto temp_2d = temp.to(torch::kCPU).to(torch::kFloat64).contiguous();
  auto conc_3d = conc.to(torch::kCPU).to(torch::kFloat64).contiguous();
  auto flux = band_flux.to(torch::kCPU).to(torch::kFloat64).contiguous();

  TORCH_CHECK(dz_1d.dim() == 1, "dz must be 1D (nlyr)");
  TORCH_CHECK(pres_2d.dim() == 2, "pres must be 2D (ncol, nlyr)");
  TORCH_CHECK(temp_2d.dim() == 2, "temp must be 2D (ncol, nlyr)");
  TORCH_CHECK(conc_3d.dim() == 3, "conc must be 3D (ncol, nlyr, nspecies)");
  TORCH_CHECK(flux.dim() == 3, "band_flux must be 3D (ncol, nlyr+1, 2)");
  TORCH_CHECK(pres_2d.size(0) == 1 && temp_2d.size(0) == 1 &&
                  conc_3d.size(0) == 1 && flux.size(0) == 1,
              "write_column_profile currently expects ncol = 1");

  int64_t const nlyr = dz_1d.size(0);
  TORCH_CHECK(pres_2d.size(1) == nlyr, "pres size mismatch with dz");
  TORCH_CHECK(temp_2d.size(1) == nlyr, "temp size mismatch with dz");
  TORCH_CHECK(conc_3d.size(1) == nlyr, "conc size mismatch with dz");
  TORCH_CHECK(flux.size(1) == nlyr + 1, "band_flux must have nlyr+1 levels");
  TORCH_CHECK(flux.size(2) == 2, "band_flux last dimension must be 2");
  TORCH_CHECK(conc_3d.size(2) == static_cast<int64_t>(species_names.size()),
              "conc species dimension must match species_names");

  std::filesystem::create_directories(filename.parent_path());
  std::ofstream out(filename);
  TORCH_CHECK(out, "Failed to open output file: ", filename.string());

  auto mmw = mean_molecular_weight(conc_3d).squeeze(0).contiguous();
  auto total_conc = conc_3d.sum(-1).squeeze(0).contiguous();
  auto upward = flux.index({0, torch::indexing::Slice(), 0}).contiguous();
  auto downward = flux.index({0, torch::indexing::Slice(), 1}).contiguous();
  auto net = upward - downward;
  int64_t const nspecies_out = conc_3d.size(2) - 1;
  int const idx_width = 6;
  int const value_width = 16;

  std::vector<double> level_height_km(static_cast<size_t>(nlyr), 0.0);
  double cumulative_km = 0.0;
  for (int64_t i = 0; i < nlyr; ++i) {
    level_height_km[static_cast<size_t>(i)] = cumulative_km;
    cumulative_km += dz_1d[i].item<double>() * 1.0e-3;
  }

  out << "# IDX: layer index (1-based); final separated row is TOA boundary\n";
  out << "# HGT: bottom-of-layer level height [km]\n";
  out << "# PRE: layer pressure [bar]\n";
  out << "# TEM: layer temperature [K]\n";
  out << "# species fields: mole fraction [ppmv], first species omitted\n";
  out << "# MMW: mean molecular weight [g/mol]\n";
  out << "# FUP: upward flux [W/m^2]\n";
  out << "# FDN: downward flux [W/m^2]\n";
  out << "# FNT: net flux = FUP - FDN [W/m^2]\n";

  out << std::setw(idx_width) << "IDX" << std::setw(value_width) << "HGT"
      << std::setw(value_width) << "PRE" << std::setw(value_width) << "TEM";
  for (int64_t is = 1; is < conc_3d.size(2); ++is) {
    auto const& name = species_names[static_cast<size_t>(is)];
    out << std::setw(
               std::max<int>(value_width, static_cast<int>(name.size()) + 2))
        << name;
  }
  out << std::setw(value_width) << "MMW" << std::setw(value_width) << "FUP"
      << std::setw(value_width) << "FDN" << std::setw(value_width) << "FNT"
      << "\n";

  out << std::setprecision(8);
  for (int64_t i = 0; i < nlyr; ++i) {
    out << std::setw(idx_width) << (i + 1);
    out << std::setw(value_width) << std::fixed
        << level_height_km[static_cast<size_t>(i)];
    out << std::setw(value_width) << std::defaultfloat
        << pres_2d[0][i].item<double>() * 1.0e-5;
    out << std::setw(value_width) << temp_2d[0][i].item<double>();

    auto total = total_conc[i].item<double>();
    for (int64_t is = 1; is < conc_3d.size(2); ++is) {
      double ppmv = 0.0;
      if (total > 0.0) {
        ppmv = conc_3d[0][i][is].item<double>() / total * 1.0e6;
      }
      auto const& name = species_names[static_cast<size_t>(is)];
      out << std::setw(
                 std::max<int>(value_width, static_cast<int>(name.size()) + 2))
          << ppmv;
    }

    out << std::setw(value_width) << mmw[i].item<double>() * 1.0e3;
    out << std::setw(value_width) << upward[i].item<double>();
    out << std::setw(value_width) << downward[i].item<double>();
    out << std::setw(value_width) << net[i].item<double>() << "\n";
  }

  out << std::setw(idx_width) << (nlyr + 1);
  out << std::setw(value_width) << std::fixed << cumulative_km;
  out << std::setw(value_width) << std::defaultfloat
      << (pres_2d[0][nlyr - 1].item<double>() * 1.0e-5);
  out << std::setw(value_width) << temp_2d[0][nlyr - 1].item<double>();
  for (int64_t is = 1; is < conc_3d.size(2); ++is) {
    double ppmv = 0.0;
    auto total = total_conc[nlyr - 1].item<double>();
    if (total > 0.0) {
      ppmv = conc_3d[0][nlyr - 1][is].item<double>() / total * 1.0e6;
    }
    auto const& name = species_names[static_cast<size_t>(is)];
    out << std::setw(
               std::max<int>(value_width, static_cast<int>(name.size()) + 2))
        << ppmv;
  }
  out << std::setw(value_width) << mmw[nlyr - 1].item<double>() * 1.0e3;
  out << std::setw(value_width) << upward[nlyr].item<double>();
  out << std::setw(value_width) << downward[nlyr].item<double>();
  out << std::setw(value_width) << net[nlyr].item<double>() << "\n";
}

}  // namespace harp
