#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on

// harp
#include "cdisort213/cdisort.h"

namespace harp {

//! \brief Common base class for all RT solvers
class RTSolverImpl {
 public:
  RTSolverImpl() = default;
  virtual ~RTSolverImpl() {}
  virtual torch::Tensor forward(
      torch::Tensor prop, std::map<std::string, torch::Tensor>& bc,
      torch::optional<torch::Tensor> temf = torch::nullopt) {
    throw std::runtime_error("RTSolverImpl::forward: not implemented");
  }
};

using RTSolver = std::shared_ptr<RTSolverImpl>;

struct BeerLambertOptions {
  BeerLambertOptions() = default;

  //! \note $T ~ Ts*(\tau/\tau_s)^\alpha$ scaling at lower boundary
  ADD_ARG(float, alpha);
};

class BeerLambertImpl : public torch::nn::Cloneable<BeerLambertImpl>,
                        public RTSolverImpl {
 public:
  //! options with which this `BeerLambertImpl` was constructed
  BeerLambertOptions options;

  //! Constructor to initialize the layers
  BeerLambertImpl() = default;
  explicit BeerLambertImpl(BeerLambertOptions const& options);
  void reset() override;

  //! Calculate radiative intensity
  /*!
   * \note export shared variable `radiation/<band_name>/optics`
   *
   * \param prop properties at each level (..., layer)
   * \param ftoa top of atmosphere flux
   * \param temf temperature at each level (layer + 1)
   */
  torch::Tensor forward(
      torch::Tensor prop, std::map<std::string, torch::Tensor>& bc,
      torch::optional<torch::Tensor> temf = torch::nullopt) override;
};
TORCH_MODULE(BeerLambert);

}  // namespace harp
