#pragma once

namespace harp {
enum index {
  // hydro variables
  ITM = 0,  //! temperature
  IPR = 1,  //! pressure
  ICX = 2,  //! mole fractions

  // optical variables
  IAB = 0,  //! absorption
  ISS = 1,  //! single scattering albedo
  IPM = 2,  //! phase moments

  // flux variables
  IUP = 0,  //! upward
  IDN = 1,  //! downward

  // spectral variables
  IWN = 0,  //! wavenumber(length)
  IWT = 1,  //! weight
};

enum {
  // phase functions
  kIsotropic = 0,
  kRayleigh = 1,
  kHenyeyGreenstein = 2,
  kDoubleHenyeyGreenstein = 3,
  kHazeGarciaSiewert = 4,
  kCloudGarciaSiewert = 5,

  // interpolation orders
  k2ndOrder = 2,
  k4thOrder = 4,

  // boundary conditions
  kExtrapolate = 0,
  kConstant = 1,
};
}  // namespace harp
