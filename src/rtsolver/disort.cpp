// C/C++
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

// harp
#include <index.h>

#include <utils/vectorize.hpp>

#include "rtsolver.hpp"

namespace harp {
DisortOptions::DisortOptions() {
  ds().bc.btemp = 0.;
  ds().bc.ttemp = 0.;
  ds().bc.fluor = 0.;
  ds().bc.albedo = 0.;
  ds().bc.fisot = 0.;
  ds().bc.fbeam = 0.;
  ds().bc.temis = 0.;
  ds().bc.umu0 = 1.;
  ds().bc.phi0 = 0.;
  ds().accur = 1.E-6;
}

void DisortOptions::set_header(std::string const& header) {
  snprintf(ds().header, sizeof(ds().header), "%s", header.c_str());
}

void DisortOptions::set_flags(std::string const& str) {
  std::vector<std::string> dstr = Vectorize<std::string>(str.c_str(), " ,");

  for (int i = 0; i < dstr.size(); ++i) {
    if (dstr[i] == "ibcnd") {
      ds().flag.ibcnd = true;
    } else if (dstr[i] == "usrtau") {
      ds().flag.usrtau = true;
    } else if (dstr[i] == "usrang") {
      ds().flag.usrang = true;
    } else if (dstr[i] == "lamber") {
      ds().flag.lamber = true;
    } else if (dstr[i] == "planck") {
      ds().flag.planck = true;
    } else if (dstr[i] == "spher") {
      ds().flag.spher = true;
    } else if (dstr[i] == "onlyfl") {
      ds().flag.onlyfl = true;
    } else if (dstr[i] == "quiet") {
      ds().flag.quiet = true;
    } else if (dstr[i] == "intensity_correction") {
      ds().flag.intensity_correction = true;
    } else if (dstr[i] == "old_intensity_correction") {
      ds().flag.old_intensity_correction = true;
    } else if (dstr[i] == "general_source") {
      ds().flag.general_source = true;
    } else if (dstr[i] == "output_uum") {
      ds().flag.output_uum = true;
    } else if (dstr[i] == "print-input") {
      ds().flag.prnt[0] = true;
    } else if (dstr[i] == "print-fluxes") {
      ds().flag.prnt[1] = true;
    } else if (dstr[i] == "print-intensity") {
      ds().flag.prnt[2] = true;
    } else if (dstr[i] == "print-transmissivity") {
      ds().flag.prnt[3] = true;
    } else if (dstr[i] == "print-phase-function") {
      ds().flag.prnt[4] = true;
    } else {
      std::stringstream msg;
      msg << "flag: '" << dstr[i] << "' unrecognized" << std::endl;
      throw std::runtime_error("DisortOptions::set_flags::" + msg.str());
    }
  }
}

DisortImpl::DisortImpl(DisortOptions const& options_) : options(options_) {
  options.set_header(options.header());
  c_disort_state_alloc(&options.ds());
  c_disort_out_alloc(&options.ds(), &options.ds_out());
}

DisortImpl::~DisortImpl() {
  c_disort_state_free(&options.ds());
  c_disort_out_free(&options.ds(), &options.ds_out());
}

//! \note Counting Disort Index
//! Example, il = 0, iu = 2, ds_.nlyr = 6, partition in to 3 blocks
//! face id   -> 0 - 1 - 2 - 3 - 4 - 5 - 6
//! cell id   -> | 0 | 1 | 2 | 3 | 4 | 5 |
//! disort id -> 6 - 5 - 4 - 3 - 2 - 1 - 0
//! blocks    -> ---------       *       *
//!           ->  r = 0  *       *       *
//!           ->         ---------       *
//!           ->           r = 1 *       *
//!           ->                 ---------
//!           ->                   r = 2
//! block r = 0 gets, 6 - 5 - 4
//! block r = 1 gets, 4 - 3 - 2
//! block r = 2 gets, 2 - 1 - 0
torch::Tensor DisortImpl::forward(torch::Tensor prop, torch::Tensor ftoa,
                                  torch::Tensor temf) {
  auto& ds = options.ds();
  auto& ds_out = options.ds_out();

  if (ds.flag.ibcnd != 0) {
    throw std::runtime_error("DisortImpl::forward: ibcnd != 0");
  }

  int nwave = prop.size(1);
  int nx3 = prop.size(2);
  int nx2 = prop.size(3);
  int nx1 = prop.size(4);

  auto flx = torch::zeros({2, nwave, nx3, nx2, nx1}, torch::kDouble);

  // tensor accessors
  auto aprop = prop.accessor<double, 5>();
  auto aftoa = ftoa.accessor<double, 3>();
  auto atemf = temf.accessor<double, 3>();
  auto aflx = flx.accessor<double, 5>();

  // run disort
  int rank_in_column = 0;
  for (int k = 0; k < nx3; ++k)
    for (int j = 0; j < nx2; ++j) {
      if (ds.flag.planck) {
        for (int i = 0; i <= ds.nlyr; ++i) {
          ds.temper[ds.nlyr - i] = atemf[k][j][i];
        }
      }

      for (int n = 0; n < nwave; ++n) {
        // stellar source function
        ds.bc.fbeam = aftoa[n][k][j];

        for (int i = 0; i < ds.nlyr; ++i) {
          // absorption
          ds.dtauc[ds.nlyr - 1 - i] = aprop[IAB][n][k][j][i];

          // single scatering albedo
          ds.ssalb[ds.nlyr - 1 - i] = aprop[ISS][n][k][j][i];

          // Legendre coefficients
          for (int m = 0; m <= ds.nmom; ++m)
            ds.pmom[(ds.nlyr - 1 - i) * (ds.nmom + 1) + m] =
                aprop[IPM + m][n][k][j][i];
        }

        c_disort(&ds, &ds_out);

        for (int i = 0; i <= ds.nlyr; ++i) {
          int i1 = ds.nlyr - (rank_in_column * (ds.nlyr - 1) + i);
          aflx[IUP][n][k][j][i] = ds_out.rad[i1].flup;
          aflx[IDN][n][k][j][i] = ds_out.rad[i1].rfldir + ds_out.rad[i1].rfldn;
        }
      }
    }

  return flx;
}
}  // namespace harp
