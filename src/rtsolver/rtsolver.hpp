#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <harp/add_arg.h>

namespace harp {

struct BeerLambertOptions {
  BeerLambertOptions() = default;

  //! \note $T ~ Ts*(\tau/\tau_s)^\alpha$ scaling at lower boundary
  ADD_ARG(float, alpha);
};

class BeerLambertImpl : public torch::nn::Cloneable<BeerLambertImpl> {
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
   * \param prop properties at each level (..., nlyr)
   * \param ftoa top of atmosphere flux
   * \param temf temperature at each level (..., nlyr+1)
   */
  torch::Tensor forward(torch::Tensor prop,
                        std::map<std::string, torch::Tensor>* bc,
                        torch::optional<torch::Tensor> temf = torch::nullopt);
};
TORCH_MODULE(BeerLambert);

}  // namespace harp

#undef ADD_ARG
