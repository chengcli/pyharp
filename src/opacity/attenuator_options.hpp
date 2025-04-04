#pragma once

// harp
#include <harp/add_arg.h>

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

  //! opacity scale
  ADD_ARG(double, scale) = 1.0;

  //! metallicity (used in Freedman mean opacities)
  ADD_ARG(double, metallicity) = 0.0;

  //! kappa_a (used in xiz semigrey opacity)
  ADD_ARG(double, kappa_a) = 0.0;

  //! kappa_b (used in xiz semigrey opacity)
  ADD_ARG(double, kappa_b) = 0.0;

  //! kappa_cut (used in xiz semigrey opacity)
  ADD_ARG(double, kappa_cut) = 0.0;

  //! number of scattering moments
  ADD_ARG(int, nmom) = 0;
};

}  // namespace harp
