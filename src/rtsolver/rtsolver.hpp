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
  virtual torch::Tensor forward(torch::Tensor prop, torch::Tensor ftoa,
                                torch::Tensor temf) {
    throw std::runtime_error("RTSolverImpl::forward: not implemented");
  }
};

using RTSolver = std::shared_ptr<RTSolverImpl>;

struct DisortOptions {
  DisortOptions();
  void set_header(std::string const &header);
  void set_flags(std::string const &flags);

  // header
  ADD_ARG(std::string, header) = "running disort ...";
  ADD_ARG(std::string, flags) = "";

  // spectral dimensions
  ADD_ARG(int, nwave) = 1;

  // placeholder for disort state
  ADD_ARG(disort_state, ds);
};

class DisortImpl : public torch::nn::Cloneable<DisortImpl>,
                   public RTSolverImpl {
 public:
  //! options with which this `DisortImpl` was constructed
  DisortOptions options;

  std::vector<disort_state> ds;
  std::vector<disort_output> ds_out;

  //! Constructor to initialize the layers
  DisortImpl() = default;
  explicit DisortImpl(DisortOptions const &options);
  virtual ~DisortImpl();
  void reset() override;

  //! Calculate radiative flux or intensity
  /*!
   * \param prop properties at each level (nwave, ncol, nlyr, nprop)
   * \param ftoa top of atmosphere flux (nwave, ncol)
   * \param temf temperature at each level (ncol, nlvl = nlyr + 1)
   * \return radiative flux or intensity (nwave, ncol, nlvl, 2)
   */
  torch::Tensor forward(torch::Tensor prop, torch::Tensor ftoa,
                        torch::Tensor temf) override;

 private:
  bool allocated_ = false;
};
TORCH_MODULE(Disort);

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
  explicit BeerLambertImpl(BeerLambertOptions const &options);
  void reset() override;

  //! Calculate radiative intensity
  /*!
   * \note export shared variable `radiation/<band_name>/optics`
   *
   * \param prop properties at each level (..., layer)
   * \param ftoa top of atmosphere flux
   * \param temf temperature at each level (layer + 1)
   */
  torch::Tensor forward(torch::Tensor prop, torch::Tensor ftoa,
                        torch::Tensor temf) override;
};
TORCH_MODULE(BeerLambert);

}  // namespace harp
