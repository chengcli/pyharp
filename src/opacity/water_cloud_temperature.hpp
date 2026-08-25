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

//! Temperature-selected liquid-water or water-ice cloud opacity.
class TemperatureSwitchWaterCloudImpl
    : public torch::nn::Cloneable<TemperatureSwitchWaterCloudImpl> {
 public:
  //! Fixed phase-transition temperature used until microphysics supplies phase.
  static constexpr double kFreezingTemperature = 273.15;  // K

  //! Options with which this module was constructed.
  OpacityOptions options;

  //! Constituent optical-property models.
  FuWaterIce ice{nullptr};
  MieWaterLiquid liquid{nullptr};

  TemperatureSwitchWaterCloudImpl() : options(OpacityOptionsImpl::create()) {}
  explicit TemperatureSwitchWaterCloudImpl(OpacityOptions const& options_);
  void reset() override;

  //! Calculate cloud properties using temperature to select condensate phase.
  /*!
   * The configured species supplies the total water-condensate concentration.
   * At each layer, Fu (1996, 1998) ice optics are used for temp < 273.15 K;
   * Lorenz-Mie liquid-water optics are used for temp >= 273.15 K. The phase
   * boundary is deliberately fixed while SNAPy does not output H2O(l) and
   * H2O(s) separately.
   *
   * \param conc molar concentration [mol/m^3], (ncol, nlyr, nspecies)
   * \param kwargs must contain `temp` [K], `re` [um], and either
   *        `wavenumber` [cm^-1] or `wavelength` [um]
   * \return optical properties, (nwave, ncol, nlyr, 2 + nmom)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(TemperatureSwitchWaterCloud);

}  // namespace harp
