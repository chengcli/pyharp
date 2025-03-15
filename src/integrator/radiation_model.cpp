// harp
#include "radiation_model.hpp"

#include <constants.h>

#include <calc_dz_hypsometric.hpp>

namespace harp {

//! dump of shared data to other modules
extern std::unordered_map<std::string, torch::Tensor> shared;

RadiationModelImpl::RadiationModelImpl(RadiationModelOptions const& options_)
    : options(options_) {
  // set up integrator
  pintg = register_module("intg", Integrator(options.intg()));

  // set up radiation model
  prad = register_module("rad", Radiation(options.rad()));

  // set up stage registers
  atemp0_ = register_buffer(
      "atemp0",
      torch::zeros({options.ncol(), options.nlyr()}, torch::kFloat64));
  atemp1_ = register_buffer(
      "atemp1",
      torch::zeros({options.ncol(), options.nlyr()}, torch::kFloat64));
  btemp0_ = register_buffer("btemp0",
                            torch::zeros({options.ncol()}, torch::kFloat64));
  btemp1_ = register_buffer("btemp1",
                            torch::zeros({options.ncol()}, torch::kFloat64));
}

int RadiationModel::forward(torch::Tensor xfrac,
                            std::map<std::string, torch::Tensor>& atm,
                            std::map<std::string, torch::Tensro>& bc, double dt,
                            int stage) {
  // -------- (1) save initial state --------
  if (stage == 0) {
    atemp0_.copy_(atm["temp"]);
    atemp1_.copy_(atm["temp"]);
    btemp0_.copy_(bc["btemp"]);
    btemp1_.copy_(bc["btemp"]);
  }

  auto dz =
      calc_dz_hypsometric(atm["pres"], atm["temp"],
                          torch::tensor({options.mean_mol_weight() *
                                         options.grav() / Constants::Rgas}));

  // -------- (2) run one time step --------
  auto conc = torch::empty_like(xfrac);

  conc.select(-1, 0) = mean_co2 * (atm["pres"] / (R * atm["temp"]));
  conc.select(-1, 1) = mean_h2o * (atm["pres"] / (R * atm["temp"]));
  conc.select(-1, 2) = mean_so2 * (atm["pres"] / (R * atm["temp"]));

  // aerosols
  conc.narrow(-1, 3, new_X.size(-1)) =
      options.aero_scale() * new_X.unsqueeze(0) * atm["pres"].unsqueeze(-1) /
      (constants::Rgas * atm["temp"].unsqueeze(-1));

  auto netflux = prad->forward(conc, dz, &bc, &atm);
  shared["result/netflux"] = netflux;

  auto surf_forcing =
      shared["radiation/downward_flux"] - 5.67e-8 * bc["btemp"].pow(4);

  auto dT_surf = surf_forcing * (tstep / options.cSurf());
  shared["result/dT_surf"] = dT_surf;

  // unit = [kg/m^3]
  auto rho = (atm["pres"] * mean_mol_weight) / (R * atm["temp"]);
  auto dT_atm = -tstep / (rho * options.cp() * dz) *
                (netflux.narrow(-1, 1, nlyr) - netflux.narrow(-1, 0, nlyr));
  shared["result/dT_atm"] = dT_atm;

  // -------- (3) multi-stage averaging --------
  atm["temp"].copy_(pintg->forward(stage, atemp0_, atemp1_, dT_atm));
  atm["temp"].clamp_(20, 1000);
  atm1_.copy_(atm["temp"]);

  bc["btemp"].copy_(pintg->forward(stage, btemp0_, btemp1_, dT_surf));
  bc["btemp"].clamp_(20, 1000);
  btemp1_.copy_(bc["btemp"]);

  return 0;
}

}  // namespace harp
