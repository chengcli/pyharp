#pragma once

// C/C++
#include <map>
#include <string>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

// harp
#include "opacity_options.hpp"

namespace harp {

//! Gas Rayleigh-scattering opacity on an arbitrary spectral grid.
class RayleighImpl : public torch::nn::Cloneable<RayleighImpl> {
 public:
  //! Multiplicative cross-section scale for every configured species.
  torch::Tensor species_scales;

  //! Options with which this module was constructed.
  OpacityOptions options;

  RayleighImpl() : options(OpacityOptionsImpl::create()) {}
  explicit RayleighImpl(OpacityOptions const& options_);
  void reset() override;

  //! Calculate Rayleigh optical properties.
  /*!
   * The concentration tensor uses molar concentration [mol/m^3]. The returned
   * tensor contains attenuation [1/m], single-scattering albedo, and Legendre
   * phase-function moments excluding the zeroth moment.
   *
   * Supported species are H2, He, H2O, CH4, N2, and CO2. H2 uses the
   * Dalgarno & Williams (1962) expression. Other species are approximated
   * by multiplying the H2 cross section by the constant scale factors
   * given in Pierrehumbert (2010).
   *
   * \param conc mole concentration [mol/m^3], (ncol, nlyr, nspecies)
   * \param kwargs must contain either "wavenumber" [cm^-1] or
   *        "wavelength" [um], both with shape (nwave)
   * \return optical properties, (nwave, ncol, nlyr, 2 + nmom)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(Rayleigh);

}  // namespace harp
