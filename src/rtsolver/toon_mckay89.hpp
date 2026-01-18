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

struct ToonMcKay89OptionsImpl {
  ToonMcKay89Options();
  static std::shared_ptr<ToonMcKay89OptionsImpl> create() {
    return std::make_shared<ToonMcKay89OptionsImpl>();
  }

  void report(std::ostream& os) const {
    os << "* zenith_correction = " << zenith_correction() << "\n";

    os << "* wave_lower = ";
    for (auto const& v : wave_lower()) os << v << ", ";
    os << "\n";

    os << "* wave_upper = ";
    for (auto const& v : wave_upper()) os << v << ", ";
    os << "\n";
  }

  //! set lower wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_lower) = {};

  //! set upper wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_upper) = {};

  //! zenith correction
  ADD_ARG(bool, zenith_correction) = false;
};

using ToonMcKay89Options = std::shared_ptr<ToonMcKay89Impl>;

class ToonMcKay89Impl : public torch::nn::Cloneable<ToonMcKay89Impl> {
 public:
  //! options with which this `ToonMcKay89Impl` was constructed
  ToonMcKay89Options options;

  //! Constructor to initialize the layers
  ToonMcKay89Impl() : options(ToonMcKay89OptionsImpl::create()) {}
  explicit ToonMcKay89Impl(ToonMcKay89Options const& options);
  void reset() override;

  //! Calculate radiative flux
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

#undef ADD_ARG
