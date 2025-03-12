#pragma once

// harp
#include <add_arg.h>

namespace harp {

struct AttenuatorOptions {
  ADD_ARG(std::string, type) = "";
  ADD_ARG(std::vector<std::string>, opacity_files) = {};
  ADD_ARG(std::vector<int>, species_ids) = { 0 };
};

}  // namespace harp
