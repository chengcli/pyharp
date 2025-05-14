#pragma once

// C/C++
#include <fstream>
#include <string>
#include <vector>

// torch
#include <torch/script.h>
#include <torch/torch.h>

namespace harp {

//! Read a variable in a "pt" file to a vector
template <typename T>
std::vector<T> read_var_pt(std::string const& fname, std::string const& vname) {
  // Check if the file exists
  std::ifstream file(fname);
  TORCH_CHECK(file.is_open(), "File ", fname, " not found");

  // Load the file
  torch::jit::script::Module container = torch::jit::load(fname);

  auto value = container.attr(vname).toTensor();

  const T* data = value.data_ptr<T>();
  return std::vector<T>(data, data + value.numel());
}

}  // namespace harp
