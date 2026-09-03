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

//! Fu (1996) solar and Fu et al. (1998) infrared water-ice opacity.
class FuWaterIceImpl : public torch::nn::Cloneable<FuWaterIceImpl> {
 public:
  //! Fu96 wavelength band edges [um] and polynomial coefficients.
  torch::Tensor fu96_wavelength_edges;
  torch::Tensor fu96_extinction_coeff;
  torch::Tensor fu96_coalbedo_coeff;
  torch::Tensor fu96_asymmetry_coeff;
  torch::Tensor fu96_delta_coeff;

  //! Fu98 wavelength nodes [um] and polynomial coefficients.
  torch::Tensor fu98_wavelength;
  torch::Tensor fu98_extinction_coeff;
  torch::Tensor fu98_absorption_coeff;
  torch::Tensor fu98_asymmetry_coeff;

  //! Options with which this module was constructed.
  OpacityOptions options;

  FuWaterIceImpl() : options(OpacityOptionsImpl::create()) {}
  explicit FuWaterIceImpl(OpacityOptions const& options_);
  void reset() override;

  //! Calculate water-ice cloud optical properties.
  /*!
   * `conc` is molar concentration [mol/m^3]. The configured water-ice
   * species is converted to ice-water content [g/m^3] using its molecular
   * weight. If effective radius re [um] is supplied, it is converted
   * internally using Fu (1996), Eq. (3.12), Dge = 8 re / (3 sqrt(3)). If re
   * is absent, Dge is diagnosed from layer temperature and IWC following Sun
   * and Rikus (1999), with the correction from Sun (2001), and limited to the
   * Fu96/Fu98 common valid interval [18.63, 129.6] um.
   * Fu96 is used below 4 um and Fu98 from 4 to 100 um.
   * All returned optical properties are zero outside 0.25--100 um.
   * Set scalar `fu_delta_scale` to true to apply the Fu96 delta scaling. It
   * defaults to false because the PyHARP Toon shortwave solver already
   * applies delta-Eddington scaling.
   *
   * The returned tensor contains attenuation [1/m], single-scattering
   * albedo, and Henyey-Greenstein Legendre moments excluding beta_0.
   *
   * \param conc molar concentration [mol/m^3], (ncol, nlyr, nspecies)
   * \param kwargs must contain `wavenumber` [cm^-1] or `wavelength` [um].
   *        Layer `re` [um] takes precedence when present; otherwise layer
   *        `temp` [K] is required for the Sun--Rikus/Sun diagnostic. Optional
   *        scalar `fu_delta_scale` selects Fu96 delta scaling.
   * \return optical properties, (nwave, ncol, nlyr, 2 + nmom)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(FuWaterIce);

}  // namespace harp
