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

class S8FullerImpl : public torch::nn::Cloneable<S8FullerImpl> {
 public:
  //! wavelength [um]
  //! (nwave, 1)
  torch::Tensor kwave;

  //! extinction x-section [m^2/mol] + single scattering albedo
  //! (nwave, nprop=2)
  torch::Tensor kdata;

  //! options with which this `S8FullerImpl` was constructed
  AttenuatorOptions options;

  //! Constructor to initialize the layer
  S8FullerImpl() = default;
  explicit S8FullerImpl(AttenuatorOptions const& options_);
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
   *          (1) <band_name> + "/wavelength": wavelength [um] (nwave)
   *          (2) <band_name> + "/wavenumber": wavenumber [1/cm] (nwave)
   *          (3) "wavelength": wavelength [um] (nwave)
   *          (4) "wavenumber": wavenumber [1/cm] (nwave)
   *        If none of the keys are found, an error is thrown.
   *
   * \return optical properties (nwave, ncol, nlyr, nprop=2)
   */
  torch::Tensor forward(torch::Tensor conc,
                        std::map<std::string, torch::Tensor> const& kwargs);
};
TORCH_MODULE(S8Fuller);

}  // namespace harp
