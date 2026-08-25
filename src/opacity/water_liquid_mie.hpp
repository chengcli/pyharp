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

//! Lorenz-Mie optical properties for spherical liquid-water droplets.
class MieWaterLiquidImpl : public torch::nn::Cloneable<MieWaterLiquidImpl> {
 public:
  //! Segelstein liquid-water refractive-index table: wavelength [um], n, k.
  torch::Tensor segelstein_water;

  //! Options with which this module was constructed.
  OpacityOptions options;

  MieWaterLiquidImpl() : options(OpacityOptionsImpl::create()) {}
  explicit MieWaterLiquidImpl(OpacityOptions const& options_);
  void reset() override;

  //! Calculate liquid-water cloud optical properties.
  /*!
   * `conc` is molar concentration [mol/m^3]. The configured H2O(l) species
   * is converted to liquid-water content [kg/m^3] using its molecular
   * weight. `re` is the radius [um] of the equivalent monodisperse spherical
   * droplets. Extinction and scattering efficiencies and the asymmetry
   * factor are evaluated from the full Lorenz-Mie series. Legendre moments
   * are represented by a Henyey-Greenstein phase function with the exact Mie
   * asymmetry factor.
   *
   * By default, the complex refractive index is log-wavelength interpolated
   * from Segelstein (1981) for liquid water at 25 C over 0.1--1000 um.
   * Callers may override it with `refractive_index_real` and
   * `refractive_index_imag`; the imaginary input is the nonnegative
   * absorption index k in n-i*k convention. Each override may be a scalar or
   * a one-dimensional tensor matching the spectral grid. `water_density`
   * [kg/m^3] may be scalar or (ncol,nlyr), and defaults to 997 kg/m^3.
   *
   * This implementation performs the Mie recurrence on CPU and copies the
   * result back to the input tensor device. It is intended as a physical
   * opacity calculation, not a differentiable Torch operation.
   *
   * \param conc molar concentration [mol/m^3], (ncol, nlyr, nspecies)
   * \param kwargs must contain `wavenumber` [cm^-1] or `wavelength` [um]
   *        and layer `re` [um]
   * \return optical properties, (nwave, ncol, nlyr, 2 + nmom)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(MieWaterLiquid);

}  // namespace harp
