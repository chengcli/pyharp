// harp
#include "toon_mckay89.hpp"

#include <harp/radiation/bbflux.hpp>

#include "rtsolver_dispatch.hpp"

namespace harp {

ToonMcKay89Impl::ToonMcKay89Impl(ToonMcKay89Options const& options)
    : options(options) {
  reset();
}

void ToonMcKay89Impl::reset() {
  // No parameters to initialize
}

torch::Tensor ToonMcKay89Impl::forward(torch::Tensor prop,
                                       std::map<std::string, torch::Tensor>* bc,
                                       std::string bname,
                                       torch::optional<torch::Tensor> temf) {
  // check dimensions
  TORCH_CHECK(prop.dim() == 4, "ToonMcKay89::forward: prop.dim() != 4");

  int nwave = prop.size(0);
  int ncol = prop.size(1);
  int nlyr = prop.size(2);

  // optical thickness
  auto tau = prop.select(-1, 0).flip(-1);

  // single scattering albedo
  auto w0 = prop.select(-1, 1).flip(-1);

  // scattering asymmetry parameter
  auto g = prop.select(-1, 2).flip(-1);

  // add slash
  if (bname.size() > 0 && bname.back() != '/') {
    bname += "/";
  }

  // check bc
  if (bc->find(bname + "umu0") != bc->end()) {
    TORCH_CHECK(bc->at(bname + "umu0").dim() == 1,
                "ToonMcKay89::forward: bc->umu0.dim() != 1");
    TORCH_CHECK(bc->at(bname + "umu0").size(0) == ncol,
                "ToonMcKay89::forward: bc->umu0.size(0) != ncol");
    (*bc)["umu0"] = bc->at(bname + "umu0");
  } else {
    (*bc)["umu0"] = torch::ones({1, ncol}, prop.options());
  }

  if (bc->find(bname + "fbeam") != bc->end()) {
    TORCH_CHECK(bc->at(bname + "fbeam").dim() == 2,
                "ToonMcKay89::forward: bc->fbeam.dim() != 2");
    TORCH_CHECK(bc->at(bname + "fbeam").size(0) == nwave,
                "ToonMcKay89::forward: bc->fbeam.size(0) != nwave");
    TORCH_CHECK(bc->at(bname + "fbeam").size(1) == ncol,
                "ToonMcKay89::forward: bc->fbeam.size(1) != ncol");
    (*bc)["fbeam"] = bc->at(bname + "fbeam");
  } else {
    (*bc)["fbeam"] = torch::zeros({nwave, ncol}, prop.options());
  }

  if (bc->find(bname + "albedo") != bc->end()) {
    TORCH_CHECK(bc->at(bname + "albedo").dim() == 2,
                "ToonMcKay89::forward: bc->albedo.dim() != 2");
    TORCH_CHECK(bc->at(bname + "albedo").size(0) == nwave,
                "ToonMcKay89::forward: bc->albedo.size(0) != nwave");
    TORCH_CHECK(bc->at(bname + "albedo").size(1) == ncol,
                "ToonMcKay89::forward: bc->albedo.size(1) != ncol");
    (*bc)["albedo"] = bc->at(bname + "albedo");
  } else {
    (*bc)["albedo"] = torch::zeros({nwave, ncol}, prop.options());
  }

  auto flx = torch::zeros({nwave, ncol, nlyr + 1, 2}, prop.options());

  if (!temf.has_value()) {  // shortwave
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .check_all_same_dtype(true)
                    .declare_static_shape({nwave, ncol, nlyr + 1, 2},
                                          /*squash_dims=*/{2, 3})
                    .add_output(flx)
                    .add_input(prop)
                    .add_owned_input(bc->at("umu0")
                                         .view({1, ncol, 1, 1})
                                         .expand({nwave, ncol, nlyr + 1, 1})
                                         .contiguous())
                    .add_owned_input(bc->at("fbeam").view({nwave, ncol, 1, 1}))
                    .add_owned_input(bc->at("albedo").view({nwave, ncol, 1, 1}))
                    .build();

    at::native::call_toon89_sw(flx.device().type(), iter);
    return flx;
  } else {  // longwave
    /*Eigen::VectorXd temp(nlay + 1);
    Eigen::VectorXd be(nlay + 1);
    for (int i = 0; i < nlay + 1; ++i) {
      temp(i) = ds_.temper[i];
      be(i) = BB_integrate(ds_.temper[i], spec.wav1, spec.wav2);
    }*/
    auto wave_lo = torch::tensor(options->wave_lower(), prop.options())
                       .unsqueeze(-1)
                       .unsqueeze(-1);
    auto wave_hi = torch::tensor(options->wave_upper(), prop.options())
                       .unsqueeze(-1)
                       .unsqueeze(-1);

    auto be = bbflux_wavenumber(wave_lo, wave_hi, temf.value());

    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .check_all_same_dtype(true)
                    .declare_static_shape({nwave, ncol, nlyr + 1, 2},
                                          /*squash_dims=*/{2, 3})
                    .add_output(flx)
                    .add_input(prop)
                    .add_input(be)
                    .add_owned_input(bc->at("albedo").view({nwave, ncol, 1, 1}))
                    .build();

    at::native::call_toon89_lw(flx.device().type(), iter);
    return flx;
  }
}

}  // namespace harp
