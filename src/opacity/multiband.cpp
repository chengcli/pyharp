// C/C++
#include <fstream>
#include <iostream>

// torch
#include <torch/script.h>
#include <torch/torch.h>

// base
#include <configure.h>

// harp
#include <harp/constants.h>

#include <harp/math/interpolation.hpp>
#include <harp/utils/find_resource.hpp>

#include "multiband.hpp"

namespace harp {

MultiBandImpl::MultiBandImpl(AttenuatorOptions const& options_)
    : options(options_) {
  TORCH_CHECK(options.opacity_files().size() == 1,
              "Only one opacity file is allowed");

  TORCH_CHECK(options.species_ids().size() == 1, "Only one species is allowed");

  TORCH_CHECK(options.species_ids()[0] >= 0,
              "Invalid species_id: ", options.species_ids()[0]);

  TORCH_CHECK(options.type().empty() ||
                  (options.type().compare(0, 9, "multiband") == 0),
              "Mismatch opacity type: ", options.type(),
              " expecting 'multiband'");

  reset();
}

void MultiBandImpl::reset() {
  auto full_path = find_resource(options.opacity_files()[0]);

  // Load the file
  torch::jit::script::Module container = torch::jit::load(full_path);

  kwave = container.attr("wavenumber").toTensor();
  klnp = container.attr("pres").toTensor().log_();
  ktemp = container.attr("temp").toTensor();
  kdata = container.attr("kappa").toTensor().unsqueeze(-1);
  weights = container.attr("weights").toTensor();

  // register all buffers
  register_buffer("kwave", kwave);
  register_buffer("klnp", klnp);
  register_buffer("ktemp", ktemp);
  register_buffer("kdata", kdata);
  register_buffer("weights", weights);
}

torch::Tensor MultiBandImpl::forward(
    torch::Tensor conc, std::map<std::string, torch::Tensor> const& kwargs) {
  int nwave = kwave.size(0);
  int ncol = conc.size(0);
  int nlyr = conc.size(1);

  TORCH_CHECK(kwargs.count("pres") > 0, "pres is required in kwargs");
  TORCH_CHECK(kwargs.count("temp") > 0, "temp is required in kwargs");

  auto const& pres = kwargs.at("pres");
  auto const& temp = kwargs.at("temp");

  TORCH_CHECK(pres.size(0) == ncol && pres.size(1) == nlyr,
              "Invalid pres shape: ", pres.sizes(),
              "; needs to be (ncol, nlyr)");
  TORCH_CHECK(temp.size(0) == ncol && temp.size(1) == nlyr,
              "Invalid temp shape: ", temp.sizes(),
              "; needs to be (ncol, nlyr)");

  auto wave = kwave.unsqueeze(-1).unsqueeze(-1).expand({nwave, ncol, nlyr});
  auto lnp = pres.log().unsqueeze(0).expand({nwave, ncol, nlyr});
  auto tempa = temp.unsqueeze(0).expand({nwave, ncol, nlyr});
  auto out = interpn({wave, lnp, tempa}, {kwave, klnp, ktemp}, kdata);

  // ln(cm^2 / molecule) -> 1/m
  return 1.e-4 * constants::Avogadro * out.exp() *
         conc.select(-1, options.species_ids()[0]).unsqueeze(0).unsqueeze(-1);
}

}  // namespace harp
