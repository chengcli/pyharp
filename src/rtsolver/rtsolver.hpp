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

struct ToonMcKay89Options {
  ToonMcKay89Options() = default;
};

class ToonMcKay89Impl : public torch::nn::Cloneable<ToonMcKay89Impl> {
 public:
  //! options with which this `ToonMcKay89Impl` was constructed
  ToonMcKay89Options options;

  //! Constructor to initialize the layers
  ToonMcKay89Impl() = default;
  explicit ToonMcKay89Impl(ToonMcKay89Options const& options);
  void reset() override;

  //! Calculate radiative flux or intensity
  /*!
   * \param prop optical properties at each level (nwave, ncol, nlyr, nprop)
   * \param bc dictionary of disort boundary conditions
   *        The dimensions of each recognized key are:
   *
   * \param bname name of the radiation band
   * \param temf temperature at each level (ncol, nlvl = nlyr + 1)
   * \return radiative flux or intensity (nwave, ncol, nlvl, nrad)
   */
  torch::Tensor forward(torch::Tensor prop,
                        std::map<std::string, torch::Tensor>* bc,
                        std::string bname = "",
                        torch::optional<torch::Tensor> temf = torch::nullopt);
};

}  // namespace harp
