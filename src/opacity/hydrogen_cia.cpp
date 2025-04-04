// C/C++
#include <fstream>
#include <iostream>

// harp
#include <harp/constants.h>

#include <harp/math/interpolation.hpp>
#include <harp/utils/fileio.hpp>
#include <harp/utils/find_resource.hpp>

#include "hydrogen_cia.hpp"

namespace harp {

HydrogenCIAImpl::HydrogenCIAImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);
}

void HydrogenCIAImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);
  TORCH_CHECK(!file_exists(full_path), "Failed to open file: ", full_path);

  auto cols = get_num_cols(full_path) - 1;
  auto rows = get_num_rows(full_path) - 1;

  std::ifstream file(full_path.c_str(), std::ios::in);
  int nwave, ntemp;

  ktemp = torch::empty({ntemp}, torch::kFloat64);
  double junk;
  file >> junk;
  for (int j = 0; j < ntemp; j++) {
    double val;
    file >> val;
    ktemp[j] = val;
  }

  kwave = torch::empty({nwave}, torch::kFloat64);
  kdata = torch::empty({nwave, ntemp, 1}, torch::kFloat64);
  for (int i = 0; i < nwave; i++) {
    double val;
    kwave[i] = val;
    for (int j = 0; j < ntemp; j++) {
      file >> val;
      kdata[i][j][0] = val;
    }
  }
  file.close();

  // register all buffers
  kwave = register_buffer("kwave", kwave);
  ktemp = register_buffer("ktemp", ktemp);
  kdata = register_buffer("kwave", kdata);
}

torch::Tensor HydrogenCIAImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  torch::Tensor wave;
  if (kwargs.count("wavenumber") > 0) {
    wave = kwargs.at("wavenumber");
  }
  if (kwargs.count("wavelength") > 0) {
    wave = 1.e4 / kwargs.at("wavelength");
  } else {
    TORCH_CHECK(false, "wavelength or wavenumber is required in kwargs");
  }

  // Check species id in range
  TORCH_CHECK(options.species_ids()[0] < conc.size(-1),
              "Invalid species_id: ", options.species_ids()[0]);

  auto x0 = conc.select(-1, options.species_ids()[0]);
  auto amagat = constants::Avogadro * x0 / constants::Lo;
  auto amagat_H2 = amagat * (1. - options.xHe());

  auto data = interpn({wave, temp}, {kwave, ktemp}, kdata);

  // 1/cm -> 1/m
  return 100. * torch::exp(-data) *
         (amagat_H2 * amagat_H2).unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
