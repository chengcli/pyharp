// harp
#include "radiation_model.hpp"

#include <harp/constants.h>

#include <harp/radiation/calc_dz_hypsometric.hpp>
#include <harp/radiation/radiation_formatter.hpp>
#include <harp/utils/layer2level.hpp>

namespace harp {

//! dump of shared data to other modules
extern std::unordered_map<std::string, torch::Tensor> shared;

RadiationModelImpl::RadiationModelImpl(RadiationModelOptions const& options_)
    : options(options_) {
  reset();
}

void RadiationModelImpl::reset() {
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

int RadiationModelImpl::forward(torch::Tensor xfrac,
                                std::map<std::string, torch::Tensor>& atm,
                                std::map<std::string, torch::Tensor>& bc,
                                double dt, int stage) {
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
                                         options.grav() / constants::Rgas}));

  // -------- (2) run one time step --------
  auto conc = xfrac.clone();
  conc.narrow(-1, 0, 3) *=
      atm["pres"].unsqueeze(-1) / (constants::Rgas * atm["temp"].unsqueeze(-1));

  // aerosols
  conc.narrow(-1, 3, 2) *= options.aero_scale() * atm["pres"].unsqueeze(-1) /
                           (constants::Rgas * atm["temp"].unsqueeze(-1));

  auto [netflux, dnflux, upflux] = prad->forward(conc, dz, &bc, &atm);
  // radiative flux
  shared["result/netflux"] = netflux;

  // add thermal diffusion flux
  auto vec = atm["temp"].sizes().vec();
  vec.back() += 1;
  auto dTdz = torch::zeros(vec, atm["temp"].options());
  dTdz.narrow(-1, 1, options.nlyr() - 1) =
      2. *
      (atm["temp"].narrow(-1, 1, options.nlyr() - 1) -
       atm["temp"].narrow(-1, 0, options.nlyr() - 1)) /
      (dz.narrow(-1, 1, options.nlyr() - 1) +
       dz.narrow(-1, 0, options.nlyr() - 1));

  auto surf_forcing = dnflux - constants::stefanBoltzmann * bc["btemp"].pow(4);
  auto dT_surf = surf_forcing * (dt / options.cSurf());
  shared["result/dT_surf"] = dT_surf;

  // unit = [kg/m^3]
  auto rho = (atm["pres"] * options.mean_mol_weight()) /
             (constants::Rgas * atm["temp"]);

  // density at levels
  Layer2LevelOptions l2l;
  l2l.order(k2ndOrder).lower(kExtrapolate).upper(kExtrapolate);
  auto rhoh = layer2level(dz, rho.log(), l2l).exp();

  // thermal diffusion flux
  auto thermal_flux = -options.kappa() * rhoh * options.cp() * dTdz;
  shared["result/thermal_diffusion_flux"] = thermal_flux;

  auto dT_atm = -dt / (rho * options.cp() * dz) *
                (netflux.narrow(-1, 1, options.nlyr()) +
                 thermal_flux.narrow(-1, 1, options.nlyr()) -
                 netflux.narrow(-1, 0, options.nlyr()) -
                 thermal_flux.narrow(-1, 0, options.nlyr()));
  shared["result/dT_atm"] = dT_atm;

  // -------- (3) multi-stage averaging --------
  atm["temp"].copy_(pintg->forward(stage, atemp0_, atemp1_, dT_atm));
  atm["temp"].clamp_(20, 1000);
  atemp1_.copy_(atm["temp"]);

  bc["btemp"].copy_(pintg->forward(stage, btemp0_, btemp1_, dT_surf));
  bc["btemp"].clamp_(20, 1000);
  btemp1_.copy_(bc["btemp"]);

  return 0;
}

}  // namespace harp
