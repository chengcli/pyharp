// C/C++
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

// torch
#include <ATen/TensorIterator.h>

// harp
#include <utils/vectorize.hpp>

#include "rtsolver.hpp"

namespace harp {
void call_disort_cpu(at::TensorIterator& iter, int rank_in_column,
                     std::vector<disort_state> &ds,
                     std::vector<disort_output> &ds_out);

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
  reset();
}

void DisortImpl::reset() {
  options.set_header(options.header());
  options.set_flags(options.flags());

  TORCH_CHECK(options.ds().nlyr > 0, "DisortImpl: ds.nlyr <= 0");
  TORCH_CHECK(options.ds().nmom >= 0, "DisortImpl: ds.nmom < 0");
  TORCH_CHECK(options.ds().nstr > 0, "DisortImpl: ds.nstr <= 0");
  TORCH_CHECK(options.ds().nphi > 0, "DisortImpl: ds.nphi <= 0");
  TORCH_CHECK(options.ds().numu > 0, "DisortImpl: ds.numu <= 0");
  TORCH_CHECK(options.ds().ntau > 0, "DisortImpl: ds.ntau <= 0");

  if (allocated_) {
    for (int i = 0; i < options.nwave(); ++i) {
      c_disort_state_free(&ds[i]);
      c_disort_out_free(&ds[i], &ds_out[i]);
    }
  }

  ds.resize(options.nwave());
  ds_out.resize(options.nwave());

  for (int i = 0; i < options.nwave(); ++i) {
    ds[i] = options.ds();
    c_disort_state_alloc(&ds[i]);
    c_disort_out_alloc(&ds[i], &ds_out[i]);
  }

  allocated_ = true;
  std::cout << "disort allocated = " << allocated_ << std::endl;
}

DisortImpl::~DisortImpl() {
  for (int i = 0; i < options.nwave(); ++i) {
    c_disort_state_free(&ds[i]);
    c_disort_out_free(&ds[i], &ds_out[i]);
  }
  allocated_ = false;
  std::cout << "disort allocated = " << allocated_ << std::endl;
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
  TORCH_CHECK(options.ds().flag.ibcnd == 0, "DisortImpl::forward: ds.ibcnd != 0");
  TORCH_CHECK(prop.dim() == 4, "DisortImpl::forward: prop.dim() != 4");

  int nwave = prop.size(0);
  int ncol = prop.size(1);
  int nlyr = prop.size(2);

  TORCH_CHECK(options.ds().nlyr == nlyr, "DisortImpl::forward: ds.nlyr != nlyr");
  TORCH_CHECK(temf.size(0) == ncol, "DisortImpl::forward: temf.size(0) != ncol");
  TORCH_CHECK(temf.size(1) == nlyr + 1, "DisortImpl::forward: temf.size(1) != nlyr + 1");

  auto flx = torch::zeros({nwave, ncol, nlyr + 1, 2}, prop.options());
  auto index = torch::range(0, nwave, 1).view({-1, 1}).expand({-1, ncol}).unsqueeze(-1).unsqueeze(-1);
  int rank_in_column = 0;

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .declare_static_shape({nwave, ncol, nlyr + 1, 2}, /*squash_dims=*/{2, 3})
          .add_output(flx)
          .add_input(prop)
          .add_owned_const_input(ftoa.unsqueeze(-1).unsqueeze(-1))
          .add_owned_const_input(temf.unsqueeze(0).expand({nwave, ncol, nlyr + 1}).unsqueeze(-1))
          .add_input(index)
          .build();

  if (prop.is_cpu()) {
    call_disort_cpu(iter, rank_in_column, ds, ds_out);
  } else if (prop.is_cuda()) {
    //call_disort_cuda(iter, rank_in_column, options.ds(), options.ds_out());
  } else {
    TORCH_CHECK(false, "DisortImpl::forward: unsupported device");
  }

  return flx;
}
}  // namespace harp
