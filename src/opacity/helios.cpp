// C/C++
#include <fstream>
#include <iostream>

// base
#include <configure.h>

// harp
#include <harp/math/interpolation.hpp>
#include <harp/utils/find_resource.hpp>

#include "helios.hpp"

namespace harp {

HeliosImpl::HeliosImpl(AttenuatorOptions const& options_) : options(options_) {
  TORCH_CHECK(options.opacity_files().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(
      options.type().empty() || (options.type().compare(0, 3, "helios") == 0),
      "Mismatch opacity type: ", options.type());

  reset();
}

void HeliosImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);

  std::ifstream file(full_path.c_str(), std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + full_path);
  }

  // skip the first line
  std::string junk_line;
  std::getline(file, junk_line);

  int ntemp, npres, nband, ng;

  // temperature, pressure, band, g-points
  file >> ntemp >> npres >> nband >> ng;

  ktemp = torch::empty({ntemp}, torch::kFloat64);
  // temperature grid
  for (int i = 0; i < ntemp; ++i) {
    double val;
    file >> val;
    ktemp[i] = val;
  }

  klnp = torch::empty({npres}, torch::kFloat64);
  // pressure grid
  for (int j = 0; j < npres; ++j) {
    double val;
    file >> val;
    klnp[j] = val;
  }
  klnp.log_();

  std::vector<double> blimits(nband + 1);
  // band limits
  for (int b = 0; b <= nband; ++b) {
    double val;
    file >> val;
    blimits[b] = val;
  }

  // g-points and weights
  kwave = torch::empty({nband * ng}, torch::kFloat64);
  for (int g = 0; g < ng; ++g) {
    double val;
    file >> val;
    for (int b = 0; b < nband; ++b)
      kwave[b * ng + g] = blimits[b] + (blimits[b + 1] - blimits[b]) * val;
  }

  for (int g = 0; g < ng; ++g) {
    double val;
    file >> val;
    weights[g] = val;
  }

  kdata = torch::empty({nband * ng, npres, ntemp, 1}, torch::kFloat64);
  for (int i = 0; i < ntemp; ++i)
    for (int j = 0; j < npres; ++j)
      for (int g = 0; g < nband * ng; ++g) {
        double val;
        file >> val;
        kdata[g][j][i][0] = val;
      }
  kdata.clamp_(1.0e-99).log_();

  file.close();

  // register all buffers
  register_buffer("kwave", kwave);
  register_buffer("klnp", klnp);
  register_buffer("ktemp", ktemp);
  register_buffer("kdata", kdata);
  register_buffer("weights", weights);
}

torch::Tensor HeliosImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int nwave = kwave.size(0);
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp").unsqueeze(0).expand({nwave, ncol, nlyr});

  TORCH_CHECK(pres.size(0) == ncol && pres.size(1) == nlyr,
              "Invalid pres shape: ", pres.sizes(),
              "; needs to be (ncol, nlyr)");
  TORCH_CHECK(temp.size(0) == ncol && temp.size(1) == nlyr,
              "Invalid temp shape: ", temp.sizes(),
              "; needs to be (ncol, nlyr)");

  auto wave = kwave.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  auto lnp = pres.log().unsqueeze(0).expand({nwave, ncol, nlyr});

  auto out = interpn({wave, lnp, temp}, {kwave, klnp, ktemp}, kdata);

  //!!! CHECK UNITS !!!!
  // ln(cm^2 / molecule) -> 1/m
  return out.exp() * conc.select(-1, 0).unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
