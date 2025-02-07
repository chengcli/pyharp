#pragma once

// harp
#include <add_arg.h>

namespace harp {

struct AttenuatorOptions {
  ADD_ARG(AtmToStandardGridOptions, atm);

  ADD_ARG(std::string, type) = "";
  ADD_ARG(std::vector<std::string>, opacity_files) = {};
  ADD_ARG(std::vector<int>, species_ids) = { 0 };

  ADD_ARG(std::vector<std::string>, species_names) = {};
  ADD_ARG(std::vector<std::string>, species_weights) = {};

  ADD_ARG(double, species_mu) = 18.e-3;  // [kg/mol]
  ADD_ARG(bool, use_wavenumber) = true;
};

}  // namespace harp
