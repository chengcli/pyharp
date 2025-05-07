#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <harp/radiation/radiation.hpp>

#include "integrator.hpp"

// arg
#include <harp/add_arg.h>

namespace harp {

struct RadiationModelOptions {
  ADD_ARG(int, ncol) = 1;                    //! number of columns
  ADD_ARG(int, nlyr) = 1;                    //! number of layers
  ADD_ARG(double, grav) = 3.711;             //! m/s^2
  ADD_ARG(double, mean_mol_weight) = 0.044;  //! kg/mol
  ADD_ARG(double, cp) = 844;                 //! J/(kg K)
  ADD_ARG(double, aero_scale) = 1.0;         //! aerosol scale factor
  ADD_ARG(double, kappa) = 2.e-5;            //! m^2/s thermal diffusivity
  ADD_ARG(double,
          cSurf) = 200000;         //! J/(m^2 K) thermal intertia of the surface
  ADD_ARG(RadiationOptions, rad);  //! radiation model options
  ADD_ARG(IntegratorOptions, intg);  //! integrator options
};

class RadiationModelImpl : public torch::nn::Cloneable<RadiationModelImpl> {
 public:
  //! options with which this `RadiationModel` was constructed
  RadiationModelOptions options;

  //! submodules
  Integrator pintg = nullptr;
  Radiation prad = nullptr;

  //! Constructor to initialize the layers
  RadiationModelImpl() = default;
  explicit RadiationModelImpl(RadiationModelOptions const& options_);
  void reset() override;

  //! Advance the atmosphere & surface temperature by one time step.
  /*!
   * This function exports the following tensor variables:
   *  - result/netflux
   *  - result/thermal_diffusion_flux
   *  - result/dT_surf
   *  - result/dT_atm
   *
   * \param xfrac species mole fraction
   * \param atm atmospheric state
   *            This normally includes the following variables:
   *              - 'temp': temperature [K], (ncol, nlyr)
   *              - 'pres': pressure [Pa], (ncol, nlyr)
   * \param bc radiation boundary conditions
   * \param dt time step
   * \param stage stage of the integrator
   * \return error code
   */
  int forward(torch::Tensor xfrac, std::map<std::string, torch::Tensor>& atm,
              std::map<std::string, torch::Tensor>& bc, double dt, int stage);

 protected:
  //! stage registers for atmospheric temperature
  torch::Tensor atemp0_, atemp1_;

  //! stage registers for surface temperature
  torch::Tensor btemp0_, btemp1_;
};

TORCH_MODULE(RadiationModel);

}  // namespace harp

#undef ADD_ARG
