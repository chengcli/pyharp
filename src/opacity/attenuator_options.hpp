#pragma once

// harp
#include <add_arg.h>

namespace harp {

struct AttenuatorOptions {
  //! type of the opacity source
  ADD_ARG(std::string, type) = "";

  //! name of the band that the opacity is associated with
  ADD_ARG(std::string, bname) = "";

  //! list of opacity data files
  ADD_ARG(std::vector<std::string>, opacity_files) = {};

  //! list of dependent species indices
  ADD_ARG(std::vector<int>, species_ids) = {};

  //! number of scattering moments
  ADD_ARG(int, nmom) = 0;
};

}  // namespace harp
