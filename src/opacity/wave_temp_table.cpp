// C/C++
#include <fstream>
#include <iostream>

// torch
#include <torch/script.h>
#include <torch/torch.h>

// harp
#include <harp/constants.h>

#include <harp/math/interpolation.hpp>
#include <harp/utils/fileio.hpp>
#include <harp/utils/find_resource.hpp>

#include "wave_temp_table.hpp"

namespace harp {

WaveTempTableImpl::WaveTempTableImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);
}

void WaveTempTableImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);

  // Load the file
  torch::jit::script::Module container = torch::jit::load(full_path);

  kwave = container.attr("wavenumber").toTensor();
  ktemp = container.attr("temp").toTensor();

  auto kdata = torch::zeros(
      {(int)options.opacity_files().size(), kwave.size(0), ktemp.size(0)},
      kwave.options());
  kdata[0] = container.attr("kappa").toTensor();

  for (int n = 1; n < options.opacity_files().size(); ++n) {
    full_path = find_resource(options.opacity_files()[n]);
    container = torch::jit::load(full_path);
    kdata[n] = container.attr("kappa").toTensor();
  }

  // register all buffers
  kwave = register_buffer("kwave", kwave);
  ktemp = register_buffer("ktemp", ktemp);
  kdata = register_buffer("kdata", kdata);
}

torch::Tensor WaveTempTableImpl::forward(
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
  auto amagat_self = amagat * options.fractions()[0];

  auto data_self = interpn({wave, temp}, {kwave, ktemp}, kdata[0]);
  auto result =
      data_self.exp() * (amagat_self * amagat_self).unsqueeze(0).unsqueeze(-1);

  for (int n = 1; n < kdata.size(0); n++) {
    auto data_other = interpn({wave, temp}, {kwave, ktemp}, kdata[n]);
    auto amagat_other = amagat * options.fractions()[n];
    result += data_other.exp() *
              (amagat_self * amagat_other).unsqueeze(0).unsqueeze(-1);
  }

  // 1/cm -> 1/m
  return 100. * result;
}

}  // namespace harp
