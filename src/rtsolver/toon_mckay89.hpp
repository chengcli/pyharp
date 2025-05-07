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

struct ToonMcKay89Options {
  ToonMcKay89Options() = default;

  //! set lower wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_lower) = {};

  //! set upper wavenumber(length) at each bin
  ADD_ARG(std::vector<double>, wave_upper) = {};

  //! zenith correction
  ADD_ARG(bool, zenith_correction) = false;
};

class ToonMcKay89Impl : public torch::nn::Cloneable<ToonMcKay89Impl> {
 public:
  //! options with which this `ToonMcKay89Impl` was constructed
  ToonMcKay89Options options;

  //! Constructor to initialize the layers
  ToonMcKay89Impl() = default;
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

 private:
  //! \brief Toon 1989 shortwave solver
  /*!
   * Based on Elsie Lee's implementation in Exo-FMS_column_ck, which was
   * based on CHIMERA code by Mike Line.
   * Ported by Xi Zhang to Eigen
   * Proted by Cheng Li to torch
   * Reference: Toon, O.B., 1989, JGR, 94,16287-16301.
   */
  torch::Tensor shortwave_solver(torch::Tensor Finc, torch::Tensor mu0,
                                 torch::Tensor dtau, torch::Tensor w0,
                                 torch::Tensor g, torch::Tensor albedo);

  //! \brief Toon 1989 longwave solver
  /*!
   * Based on Elsie Lee's implementation in Exo-FMS_column_ck, which was
   * based on CHIMERA code by Mike Line.
   * Ported by Xi Zhang to Eigen
   * Proted by Cheng Li to torch
   * Reference: Toon, O.B., 1989, JGR, 94, 16287-16301.
   */
  torch::Tensor longwave_solver(torch::Tensor be, torch::Tensor dtau,
                                torch::Tensor w0, torch::Tensor g,
                                torch::Tensor albedo);
};

}  // namespace harp

#undef ADD_ARG
