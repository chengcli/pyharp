#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <add_arg.h>

#include <radiation/radiation.hpp>

#include "integrator.hpp"

namespace harp {

struct RadiationModelOptions {
  ADD_ARG(int, ncol) = 1;                    //! number of columns
  ADD_ARG(int, nlyr) = 1;                    //! number of layers
  ADD_ARG(int, nspecies) = 5;                //! number of species
  ADD_ARG(int, nstr) = 4;                    //! number of radiation streams
  ADD_ARG(double, grav) = 3.711;             //! m/s^2
  ADD_ARG(double, mean_mol_weight) = 0.044;  //! kg/mol
  ADD_ARG(double, cp) = 844;                 //! J/(kg K)
  ADD_ARG(double, surf_sw_albedo) = 0.3;     //! surface shortwave albedo
  ADD_ARG(double, aero_scale) = 1.0;         //! aerosol scale factor
  ADD_ARG(double, sr_sun) = 2.92842e-5;      //! angular size of the sun at mars
  ADD_ARG(double, solar_temp) = 5772;        //! K
  ADD_ARG(double, lum_scale) = 0.7;          //! solar luminosity scale factor
  ADD_ARG(double,
          cSurf) = 200000;         //! J/(m^2 K) thermal intertia of the surface
  ADD_ARG(RadiationOptions, rad);  //! radiation model options
};

class RadiationModelImpl : public torch::nn::Cloneable<RadiationModelImpl> {
 public:
  //! options with which this `RadiationModel` was constructed
  IntegratorOptions options;

  //! submodules
  Integrator pintg = nullptr;
  Radiation prad = nullptr;

  //! Constructor to initialize the layers
  RadiationModelImpl() = default;
  explicit RadiationModel(RadiationModel const& options_);
  void reset() override;

  //! Advance the atmosphere & surface temperature by one time step.
  int forward(torch::Tensor conc, std::map<std::string, torch::Tensor>& atm,
              std::map<std::string, torch::Tensro>& bc, double tstep,
              int stage);

 protected:
  //! stage registers for atmospheric temperature
  torch::Tensor atemp0_, atemp1_;

  //! stage registers for surface temperature
  torch::Tensor btemp0_, btemp1_;
};

TORCH_MODULE(RadiationModelImpl);

}  // namespace harp

}  // namespace harp
