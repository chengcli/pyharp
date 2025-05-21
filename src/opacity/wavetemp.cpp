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

#include "wavetemp.hpp"

namespace harp {

WaveTempImpl::WaveTempImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(
      options.type().empty() || (options.type().compare(0, 8, "wavetemp") == 0),
      "Mismatch opacity type: ", options.type(), " expecting 'wavetemp'");

  TORCH_CHECK(options.fractions().size() == options.opacity_files().size(),
              "`fractions` and `opacity_files` must have the same size");

  reset();
}

void WaveTempImpl::reset() {
  kwave.resize(options.opacity_files().size());
  ktemp.resize(options.opacity_files().size());
  kdata.resize(options.opacity_files().size());

  auto full_path = find_resource(options.opacity_files()[0]);

  // Load the file
  torch::jit::script::Module container = torch::jit::load(full_path);

  kwave[0] = container.attr("wavenumber").toTensor();
  ktemp[0] = container.attr("temp").toTensor();
  kdata[0] = container.attr("kappa").toTensor().unsqueeze(-1);

  for (int n = 1; n < options.opacity_files().size(); ++n) {
    full_path = find_resource(options.opacity_files()[n]);
    container = torch::jit::load(full_path);
    kwave[n] = container.attr("wavenumber").toTensor();
    ktemp[n] = container.attr("temp").toTensor();
    kdata[n] = container.attr("kappa").toTensor().unsqueeze(-1);
  }

  // register all buffers
  for (int n = 0; n < kdata.size(); ++n) {
    register_buffer("kwave" + std::to_string(n), kwave[n]);
    register_buffer("ktemp" + std::to_string(n), ktemp[n]);
    register_buffer("kdata" + std::to_string(n), kdata[n]);
  }
}

torch::Tensor WaveTempImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  auto const& temp = kwargs.at("temp");

  torch::Tensor wave;
  if (kwargs.count("wavenumber") > 0) {
    wave = kwargs.at("wavenumber");
  } else if (kwargs.count("wavelength") > 0) {
    wave = 1.e4 / kwargs.at("wavelength");
  } else {
    TORCH_CHECK(false, "'wavelength' or 'wavenumber' is required in kwargs");
  }

  // Check species id in range
  TORCH_CHECK(options.species_ids()[0] < conc.size(-1),
              "Invalid species_id: ", options.species_ids()[0]);

  auto x0 = conc.select(-1, options.species_ids()[0]);
  auto amagat = constants::Avogadro * x0 / constants::Lo;
  auto amagat_self = amagat * options.fractions()[0];

  int nwave = wave.size(0);
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  auto wave1 = wave.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  auto temp1 = temp.unsqueeze(0).expand({nwave, ncol, nlyr});

  auto data_self = interpn({wave1, temp1}, {kwave[0], ktemp[0]}, kdata[0]);
  auto result =
      data_self.exp() * (amagat_self * amagat_self).unsqueeze(0).unsqueeze(-1);

  for (int n = 1; n < kdata.size(); n++) {
    nwave = kwave[n].size(0);
    auto data_other = interpn({wave1, temp1}, {kwave[n], ktemp[n]}, kdata[n]);
    auto amagat_other = amagat * options.fractions()[n];
    result += data_other.exp() *
              (amagat_self * amagat_other).unsqueeze(0).unsqueeze(-1);
  }

  // 1/cm -> 1/m
  return 100. * result;
}

}  // namespace harp
