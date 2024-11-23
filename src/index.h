#pragma once

namespace harp {
enum index {
  // hydro variables
  ITM = 0,  // temperature
  IPR = 1,  // pressure
  ICX = 2,  // composition

  // optical variables
  IAB = 0,  // absorption
  ISS = 1,  // single scattering
  IPM = 2,  // phase moments

  // flux variables
  IUP = 0,  // upward
  IDN = 1,  // downward
};

enum {
  // phase functions
  kRayleigh = 0,
  kHenyeyGreenstein = 1,
  kDoubleHenyeyGreenstein = 2,

  // interpolation orders
  k4thOrder = 10,
};
}  // namespace harp
