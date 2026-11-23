#pragma once

// C/C++
#include <map>
#include <string>
#include <utility>

// torch
#include <torch/torch.h>

namespace harp {

//! pres and temp from an opacity forward's kwargs, each checked to be
//! (ncol, nlyr) like conc (ncol, nlyr, nspecies)
inline std::pair<torch::Tensor, torch::Tensor> pres_temp_from_kwargs(
    torch::Tensor const& conc,
    std::map<std::string, torch::Tensor> const& kwargs) {
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
  return {pres, temp};
}

}  // namespace harp
