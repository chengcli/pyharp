#pragma once

// C/C++
#include <cstdlib>

// base
#include <configure.h>  // DISPATH_MACRO

// harp
#include "locate.h"

namespace harp {

/*! Multidimensional linear interpolation
 * val[0..nval-1]   : output values
 * coor[0..ndim-1]  : coordinate of the interpolation point
 * data[...]        : points to the start position of a multidimensional data
 * table. len[0..ndim-1]   : length of each dimension axis[...]        :
 * coordinates of each dimesnion is placed sequentially in axis
 */
template <int N, typename T>
DISPATCH_MACRO void interpn(T *val, T const *coor, T const *data, T const *axis,
                            size_t const *len, int ndim) {
  int i1, i2;
  i1 = locate(axis, *coor, *len);

  // if the interpolation value is out of bound
  // use the closest value
  if (i1 == -1) {
    i1 = 0;
    i2 = 0;
  } else if (i1 == *len - 1) {
    i1 = *len - 1;
    i2 = *len - 1;
  } else
    i2 = i1 + 1;

  double v1[N];
  double v2[N];

  double x1 = axis[i1];
  double x2 = axis[i2];

  if (ndim == 1) {
    for (int j = 0; j < N; ++j) {
      v1[j] = data[i1 * N + j];
      v2[j] = data[i2 * N + j];
    }
  } else {
    int s = N;
    for (int j = 1; j < ndim; ++j) s *= len[j];
    interpn<N>(v1, coor + 1, data + i1 * s, axis + *len, len + 1, ndim - 1);
    interpn<N>(v2, coor + 1, data + i2 * s, axis + *len, len + 1, ndim - 1);
  }

  if (x2 != x1)
    for (int j = 0; j < N; ++j)
      val[j] = ((*coor - x1) * v2[j] + (x2 - *coor) * v1[j]) / (x2 - x1);
  else
    for (int j = 0; j < N; ++j) val[j] = (v1[j] + v2[j]) / 2.;
}

/*! A handy function for one dimensional interpolation
 * x              : interpolation point
 * data[0..len-1] : data array
 * axis[0..len-1] : coordinates
 */
template <typename T>
T interp1(T x, T const *data, T const *axis, size_t len) {
  double value;
  interpn(&value, &x, data, axis, &len, 1, 1);
  return value;
}

}  // namespace harp
