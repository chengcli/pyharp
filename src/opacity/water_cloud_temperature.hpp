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
#include "water_ice_fu96_98.hpp"
#include "water_liquid_mie.hpp"

namespace harp {

//! Temperature-partitioned liquid-water and water-ice cloud opacity.
class TemperatureSwitchWaterCloudImpl
    : public torch::nn::Cloneable<TemperatureSwitchWaterCloudImpl> {
 public:
  //! Ice first becomes possible below this temperature.
  static constexpr double kFreezingTemperature = 273.15;  // K

  //! All condensate is ice at or below this temperature.
  static constexpr double kIceOnlyTemperature = 253.15;  // K

  //! Default liquid-droplet effective radius when liquid_re is absent.
  static constexpr double kDefaultLiquidEffectiveRadius = 14.0;  // um

  //! Options with which this module was constructed.
  OpacityOptions options;

  //! Constituent optical-property models.
  FuWaterIce ice{nullptr};
  MieWaterLiquid liquid{nullptr};

  TemperatureSwitchWaterCloudImpl() : options(OpacityOptionsImpl::create()) {}
  explicit TemperatureSwitchWaterCloudImpl(OpacityOptions const& options_);
  void reset() override;

  //! Calculate cloud properties using temperature to partition condensate.
  /*!
   * The configured species supplies the total water-condensate concentration.
   * Following Khairoutdinov and Randall (2003), condensate is entirely liquid
   * for temp >= 273.15 K, entirely ice for temp <= 253.15 K, and partitioned
   * linearly between liquid and ice at intermediate temperatures. Fu (1996,
   * 1998) supplies ice optics and Lorenz-Mie theory supplies liquid-water
   * optics. These temperature bounds are fixed while SNAPy does not output
   * H2O(l) and H2O(s) separately.
   *
   * \param conc molar concentration [mol/m^3], (ncol, nlyr, nspecies)
   * \param kwargs must contain `temp` [K] and either `wavenumber` [cm^-1] or
   *        `wavelength` [um]. Optional `ice_re` [um] takes precedence over the
   *        Sun--Rikus/Sun ice-size diagnostic. Optional `liquid_re` [um]
   *        overrides the default liquid-droplet radius of 14 um.
   * \return optical properties, (nwave, ncol, nlyr, 2 + nmom)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(TemperatureSwitchWaterCloud);

}  // namespace harp
