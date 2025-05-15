#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include "attenuator_options.hpp"

namespace harp {

class FourColumnImpl : public torch::nn::Cloneable<FourColumnImpl> {
 public:
  //! wavelength [um]
  //! (nwave, 1)
  torch::Tensor kwave;

  //! extinction x-section [m^2/mol] + single scattering albedo + g-factor (HG)
  //! (nwave, nprop=3)
  torch::Tensor kdata;

  //! options with which this `FourColumnImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  FourColumnImpl() = default;
  explicit FourColumnImpl(AttenuatorOptions const& options_);
  void reset() override;

  //! Get optical properties
  /*!
   * This function calculates the shortwave optical properties of S8
   * In the returned tensor, the first dimension is the wavelength
   * and the last dimension is the optical properties.
   * The first element of the last dimension is the extinction coefficient
   * [1/m]. The second element of the last dimension is the single scattering
   * albedo.
   * \param conc mole concentration [mol/m^3] (ncol, nlyr, nspecies)
   *
   * \param kwargs arguments for opacity calculation. It searches for
   *        a wavelength/wavenumber key in kwargs and uses it to calculate
   *        the optical properties by interpolating the data.
   *        The following is a list of possible keys in search order:
   *          (1) "wavelength": wavelength [um] (nwave)
   *          (2) "wavenumber": wavenumber [1/cm] (nwave)
   *        If none of the keys are found, an error is thrown.
   *
   * \return optical properties (nwave, ncol, nlyr, nprop=3)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(FourColumn);

}  // namespace harp
