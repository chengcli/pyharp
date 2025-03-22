// harp
#include "toon_mckay89.hpp"

#include <radiation/bbflux.hpp>

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
                                       torch::optional<torch::Tensor> temf) {
  int nlay = prop.size(-1);
  int ncol = prop.size(1);

  // optical thickness
  auto tau = prop.select(-1, 0);

  // single scattering albedo
  auto w0 = prop.select(-1, 1);

  // scattering asymmetry parameter
  auto g = prop.select(-1, 2);

  // increase the last dimension by 1 (lyr -> lvl)
  auto shape = tau.sizes().vec();
  shape.back() += 1;
  torch::Tensor tau_cum = torch::zeros(shape, tau.options());
  tau_cum.narrow(-1, 1, nlay) = torch::cumsum(tau, -1);

  auto flux = torch::zeros(shape, tau.options());

  // add slash
  if (bname.size() > 0 && bname.back() != '/') {
    bname += "/";
  }

  // Call shortwave or longwave solver based on the type of radiation
  if (!temf.has_value()) {  // shortwave
    auto Finc = bc[bname + "fbeam"];
    auto surface_albedo = bc[bname + "albedo"];
    return shortwave_solver(Finc, bc["umu0"], tau_cum, w0, g, surface_albedo);

  } else {  // longwave
    /*Eigen::VectorXd temp(nlay + 1);
    Eigen::VectorXd be(nlay + 1);
    for (int i = 0; i < nlay + 1; ++i) {
      temp(i) = ds_.temper[i];
      be(i) = BB_integrate(ds_.temper[i], spec.wav1, spec.wav2);
    }*/
    auto be = bbflux_wavenumber(wave, temp);
    auto surface_emissivity = 1. - bc[bname + "albedo"];
    return longwave_solver(be, tau_cum, w0, g, surface_emissivity);
  }
}

}  // namespace harp
