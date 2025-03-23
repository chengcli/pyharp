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

  auto prop1 = prop.flip(-1);  // from top to bottom

  // optical thickness
  auto tau = prop.select(-1, 0);

  // single scattering albedo
  auto w0 = prop.select(-1, 1);

  // scattering asymmetry parameter
  auto g = prop.select(-1, 2);

  // add slash
  if (bname.size() > 0 && bname.back() != '/') {
    bname += "/";
  }

  TORCH_CHECK(bc->count(bname + "albedo") > 0,
              "Boundary condition for surface albedo not found.");

  if (!temf.has_value()) {  // shortwave
    TORCH_CHECK(bc->count(bname + "fbeam") > 0,
                "Boundary condition for incoming flux not found.");
    TORCH_CHECK(bc->count("umu0") > 0, "Boundary condition for mu0 not found.");
    return shortwave_solver(bc->at(bname + "fbeam"), bc->at("umu0"), tau, w0, g,
                            bc->at(bname + "albedo"))
        .flip(-2);

  } else {  // longwave
    /*Eigen::VectorXd temp(nlay + 1);
    Eigen::VectorXd be(nlay + 1);
    for (int i = 0; i < nlay + 1; ++i) {
      temp(i) = ds_.temper[i];
      be(i) = BB_integrate(ds_.temper[i], spec.wav1, spec.wav2);
    }*/
    auto be = bbflux_wavenumber(wave, temp);
    return longwave_solver(be, tau, w0, g, bc->at(bname + "albedo")).flip(-2);
  }
}

}  // namespace harp
