#pragma once

// harp
#include <harp/add_arg.h>

// torch
#include <torch/nn/modules/container/any.h>

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

  //! list of kwargs to pass to the JIT module
  ADD_ARG(std::vector<std::string>, jit_kwargs) = {};

  //! opacity scale
  ADD_ARG(double, scale) = 1.0;

  //// Hydrogen Atmosphere Parameters  /////

  //! metallicity (used in Freedman mean opacities)
  ADD_ARG(double, metallicity) = 0.0;

  //// Continuum Parameters  /////

  //! number fraction of species in cia calculation
  ADD_ARG(std::vector<double>, fractions) = { 1.0 };

  //! kappa_a (used in xiz semigrey opacity)
  ADD_ARG(double, kappa_a) = 0.0;

  //! kappa_b (used in xiz semigrey opacity)
  ADD_ARG(double, kappa_b) = 0.0;

  //! kappa_cut (used in xiz semigrey opacity)
  ADD_ARG(double, kappa_cut) = 0.0;

  //// Particle Parameters  /////

  //! particle diameter in [um]
  ADD_ARG(double, diameter) = 1.0;

  //! particle extinction cross section in [cm^2]
  ADD_ARG(double, xsection) = 0.0;

  //! single scattering albedo
  ADD_ARG(double, ssa) = 0.0;

  //! fraction parameter in double Henyey-Greenstein phase function
  ADD_ARG(double, ff) = 0.0;

  //! asymmetry parameter-1 in Henyey-Greenstein phase function
  ADD_ARG(double, g1) = 0.0;

  //! asymmetry parameter-2 in Henyey-Greenstein phase function
  ADD_ARG(double, g2) = 0.0;

  //! number of scattering moments
  ADD_ARG(int, nmom) = 0;
};

}  // namespace harp

#undef ADD_ARG
