#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// arg
#include <harp/add_arg.h>

namespace harp {
struct AtmToStandardGridOptions {
  ADD_ARG(int, npres) = 1;
  ADD_ARG(int, ntemp) = 1;
  ADD_ARG(int, ncomp) = 1;
};

class AtmToStandardGridImpl
    : public torch::nn::Cloneable<AtmToStandardGridImpl> {
 public:
  //! options with which this `AtmToStandardGrid` was constructed
  AtmToStandardGridOptions options;

  //! 1D composition scale grid
  torch::Tensor xgrid;

  //! 1D temperature anomaly grid
  torch::Tensor tgrid;

  //! 2D tensor representation of reference 1d atmosphere
  /*!
   * columns : levels
   * rows    : 0 (index::ITM) -> temperature
   *           1 (index::IPR) -> pressure
   *           2 (index::ICX) -> mole fraction
   */
  torch::Tensor refatm;

  //! constructor to initialize the layer
  AtmToStandardGridImpl() = default;
  explicit AtmToStandardGridImpl(AtmToStandardGridOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override;

  //! obtain standard interpolation grid in [-1, 1] x [-1, 1] x [-1, 1]
  /*!
   * \param var_x 4D tensor representation of atmospheric variables
   *              color : 0 (index::ITM) -> temperature
   *                    : 1 (index::IPR) -> pressure
   * \param ix color index of the variable of interest in var_x
   * \return tensor of size (D, W, H, 3)
   */
  torch::Tensor forward(torch::Tensor var_x, int ix);
};
TORCH_MODULE(AtmToStandardGrid);
}  // namespace harp

#undef ADD_ARG
