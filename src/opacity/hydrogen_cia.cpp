// C/C++
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// canoe
#include <air_parcel.hpp>
#include <constants.hpp>

// climath
#include <climath/core.h>
#include <climath/interpolation.h>

// utils
#include <utils/fileio.hpp>
#include <utils/find_resource.hpp>

// opacity
#include "hydrogen_cia.hpp"

XizH2H2CIA::XizH2H2CIA() : Absorber("H2-H2-CIA") { SetPar("xHe", 0.); }

HydrogenCIAImpl::HydrogenCIAImpl() { par["xHe"] = 0.; }

HydrogenCIAImpl::HydrogenCIAImpl(AttenuatorOptions const& options_)
    : AttenuatorImpl(options_) {
  reset();
}

void HydrogenCIAImpl::reset() {
  if (!FileExists(options.opacity_file())) {
    throw std::runtime_error("opacity file not found");
  }

  cols = GetNumCols(options.opacity_file()) - 1;
  rows = GetNumRows(options.opacity_file()) - 1;

  AtmToStandardGridOptions op;
  op.ntemp(cols);
  op.npres(rows);

  sc

      kdata_h2h2 = register_buffer("kdata_h2h2",
                                   torch::zeros({cols, rows}, torch::kFloat));
  kdata_h2he =
      register_buffer("kdata_h2he", torch::zeros({cols, rows}, torch::kFloat));
}

void HydrogenCIAImpl::load() {
  if (options.opacity_file().empty()) return;

  std::string full_path = find_resource(options.opacity_file());

  std::ifstream infile(options.opacity_file(), std::ios::in);
  axis_.resize(len_[0] + len_[1]);
  kcoeff_.resize(len_[0] * len_[1]);

  double junk;
  if (infile.is_open()) {
    infile >> junk;
    for (int j = 0; j < len_[1]; j++) {
      infile >> axis_[len_[0] + j];
    }

    for (int k = 0; k < len_[0]; k++) {
      infile >> axis_[k];
      for (int j = 0; j < len_[1]; j++) infile >> kcoeff_[k * len_[1] + j];
    }

    infile.close();
  } else {
    throw RuntimeError("XizH2H2CIA::LoadCoefficient",
                       "Cannot open file: " + fname);
  }
}

torch::Tensor HydrogenCIAImpl::forward(torch::Tensor var_x) {
  namespace F = torch::nn::functional;
  auto grid = scale_grid->forward(var_x, options.var_id()[0]);

  // interpolate to model grid
  auto op = F::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
            //.align_corners(false);

            auto kcross1 = F::grid_sample(kdata_h2h2, grid, op);
  auto kcross2 = F::grid_sample(kdata_h2he, grid, op);

  auto x0 = var_x[options.var_id()[0]];
  auto xHe = par.at("xHe");
  auto amagat = x0 * var_x[index::IPR] /
                (Constants::kBoltz * var_x[index::ITM] * Constants::Lo);
  auto amagat_H2 = amagat * (1. - xHe);
  auto amatat_He = amagat * xHe;

  // 1/cm -> 1/m
  auto result1 = 100. * exp(-kcross1) * amagat_H2 * amagat_H2;
  auto result2 = 100. * exp(-kcross2) * amagat_H2 * amagat_He;

  return result1 + result2;
}
