#pragma once

// harp
#include <index.h>

#include "cdisort213/cdisort.h"

#define FLX(i, m) flx[(i) * 2 + (m)]
#define PROP(i, m) prop[(i) * nprop + (m)]
#define FTOA (*ftoa)
#define TEMF(i) temf[i]

namespace harp {

template <typename T>
void disort_impl(T* flx, T* prop, T* ftoa, T* temf, int rank_in_column,
                 disort_state& ds, disort_output& ds_out, int nprop) {
  // run disort
  if (ds.flag.planck) {
    for (int i = 0; i <= ds.nlyr; ++i) {
      ds.temper[ds.nlyr - i] = TEMF(i);
    }
  }

  // stellar source function
  ds.bc.fbeam = FTOA;

  for (int i = 0; i < ds.nlyr; ++i) {
    // absorption
    ds.dtauc[ds.nlyr - 1 - i] = PROP(i, IAB);

    // single scatering albedo
    ds.ssalb[ds.nlyr - 1 - i] = PROP(i, ISS);

    // Legendre coefficients
    ds.pmom[(ds.nlyr - 1 - i) * (ds.nmom_nstr + 1)] = 1.;
    for (int m = 0; m < nprop - 2; ++m) {
      ds.pmom[(ds.nlyr - 1 - i) * (ds.nmom_nstr + 1) + m + 1] =
          PROP(i, IPM + m);
    }

    for (int m = nprop - 2; m < ds.nmom; ++m) {
      ds.pmom[(ds.nlyr - 1 - i) * (ds.nmom_nstr + 1) + m + 1] = 0.;
    }
  }

  c_disort(&ds, &ds_out);

  for (int i = 0; i <= ds.nlyr; ++i) {
    int i1 = ds.nlyr - (rank_in_column * (ds.nlyr - 1) + i);
    FLX(i, IUP) = ds_out.rad[i1].flup;
    FLX(i, IDN) = ds_out.rad[i1].rfldir + ds_out.rad[i1].rfldn;
  }
}

}  // namespace harp

#undef FLX
#undef PROP
#undef FTOA
#undef TEMF
