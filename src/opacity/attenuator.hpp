#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
// clang-format off
#include <configure.h>
#include <add_arg.h>
#include <index.h>
// clang-format on

#include "atm_to_standard_grid.hpp"

namespace harp {
struct AttenuatorOptions {
  ADD_ARG(int, npmom) = 0;
  ADD_ARG(int, nspec) = 1;
  ADD_ARG(bool, spectral_bin) = false;
  ADD_ARG(AtmToStandardGridOptions, atm);

  ADD_ARG(std::string, type) = "";
  ADD_ARG(std::string, name) = "";
  ADD_ARG(std::string, opacity_file) = "";
  ADD_ARG(std::vector<int>, var_id) = {0};
};

//! \brief base class of all attenuators
class AttenuatorImpl {
 public:
  //! parameters for the model
  std::map<std::string, torch::Tensor> par;

  //! options with which this `Attenuator` was constructed
  AttenuatorOptions options;

  //! constructor to initialize the layer
  explicit AttenuatorImpl(AttenuatorOptions const& options_)
      : options(options_) {}
  virtual ~AttenuatorImpl() {}

  //! Get optical properties
  /*!
   * \param var_x 4D tensor representation of atmospheric variables
   *              with shape (C, D, W, H)
   *              where C is the number of variables
   *              and H is the number of vertical levels
   * \return tensor of size (batch, specs, levels, temps, comps)
   *         where the first index in batch dimension is the attenuation
   *         coefficient and the second index is the single scattering albedo
   *         and the rest of the indices are the phase function moments
   *         (excluding the zeroth moment). The units of the attenuation
   *         coefficient are [1/m] The single scattering albedo is
   *         dimensionless
   */
  virtual torch::Tensor forward(torch::Tensor var_x);
};

using Attenuator = std::shared_ptr<AttenuatorImpl>;

class AbsorberRFMImpl : public AttenuatorImpl,
                        public torch::nn::Cloneable<AbsorberRFMImpl> {
 public:
  //! extinction x-section + single scattering albedo + phase function moments
  //! (batch, specs, temps, levels, comps)
  torch::Tensor kdata;

  //! scale the atmospheric variables to the standard grid
  AtmToStandardGrid scale_grid;

  //! Constructor to initialize the layer
  explicit AbsorberRFMImpl(AttenuatorOptions const& options_)
      : AttenuatorImpl(options_) {
    reset();
  }
  void reset() override;

  //! Load opacity from data file
  virtual void load();

  //! Get optical properties
  torch::Tensor forward(torch::Tensor var_x) override;
};
TORCH_MODULE(AbsorberRFM);

}  // namespace harp
