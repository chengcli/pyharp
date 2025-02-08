#pragma once

// harp
#include <add_arg.h>

namespace harp {

struct AttenuatorOptions {
  ADD_ARG(std::string, type) = "";
  ADD_ARG(std::vector<std::string>, opacity_files) = {};
  ADD_ARG(std::vector<int>, species_ids) = { 0 };

  //! names of all species
  ADD_ARG(std::vector<std::string>, species_names) = {};

  //! molecular weights of all species [kg/mol]
  ADD_ARG(std::vector<double>, species_weights) = {};
};

}  // namespace harp
