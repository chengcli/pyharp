#include "thermodynamics.h"

#include "Constants.h"
#include "Meshgrid.h"

#define MAX_ITER 2000
#define PRECISION 1.E-8

double get_mu(AirParcel const& air) {
  double result = 0.;
  for (int i = 0; i < air.ngas(); ++i)
    result += air.gas(i)->mu() * air.mcomp(i);
  return result;
}

double get_gamma(AirParcel const& air) {
  double beta, beta1 = 0., beta2 = 0., tmol = 0.;

  for (int i = 0; i < air.ndry(); ++i) tmol += air.mgas(i);

  for (int i = 0; i < air.ncloud(); ++i) {
    std::string name = air.cloud(i)->name();
    int j = air.get_id(name.substr(0, name.size() - 3));
    if (j > 0 && air.mcloud(i) > 0.) {
      beta = 1.E3 * air.cloud(i)->latent(air.get_temp()) /
             (Constants::Rgas * air.get_temp());
      beta1 += air.mgas(j) / tmol * beta;
      beta2 += air.mgas(j) / tmol * beta * beta;
    }
  }

  return (1. + beta1) /
         (get_cpt(air) / get_gas_mols(air) / Constants::Rgas + beta2);
}

double get_gas_mols(AirParcel const& air) {
  double result = 0.;
  for (int i = 0; i < air.ngas(); ++i) result += air.mgas(i);
  return result;
}

double get_hydro_dist(AirParcel const& air1, AirParcel const& air2,
                      double grav) {
  double dlnp = log(air1.get_pa() / air2.get_pa());

  double m1 = air1.get_temp() * get_gas_mols(air1) / get_mu(air1),
         m2 = air2.get_temp() * get_gas_mols(air2) / get_mu(air2);

  return 0.5 * (m1 + m2) * Constants::Rgas / grav * dlnp;
}

double get_entropy(AirParcel const& air) {
  double tmol = get_gas_mols(air), svap = 0., rlnp = 0., latent = 0.;

  for (int i = 0; i < air.ngas(); ++i) {
    svap += air.gas(i)->entropy(air.get_temp()) * air.mcomp(i);
    /** \note when molar amount is nearly zero, skip calculate its contribution
     */
    if (!std::isinf(log(air.mgas(i))))
      rlnp += Constants::Rgas * log(air.mgas(i) * air.get_bar() / tmol) *
              air.mcomp(i);
  }

  for (int i = 0; i < air.ncloud(); ++i)
    latent += 1.E3 * air.cloud(i)->latent(air.get_temp()) / air.get_temp() *
              air.mcloud(i);

  return svap - rlnp - latent;
}

double get_enthalpy(AirParcel const& air) {
  double latent = 0., hvap = 0.;

  for (int i = 0; i < air.ngas(); ++i) {
    hvap += air.gas(i)->enthalpy(air.get_temp()) * air.mcomp(i);
  }

  for (int i = 0; i < air.ncloud(); ++i)
    latent += air.cloud(i)->latent(air.get_temp()) * air.mcloud(i);

  return hvap - latent;
}

double get_energy(AirParcel const& air) { return 0.; }

double get_cpt(AirParcel const& air) {
  double cpt = 0.;

  for (int i = 0; i < air.ngas() + air.ncloud(); ++i)
    cpt += air.gas(i)->cp(air.get_temp()) * air.mgas(i);

  return cpt;
}

double get_vtemp(AirParcel const& air) {
  double mud = 0., xd = 0.;

  for (int i = 0; i < air.ndry(); ++i) {
    mud += air.gas(i)->mu() * air.mgas(i);
    xd += air.mgas(i);
  }

  mud /= xd;

  return air.get_temp() * mud * get_gas_mols(air) / get_mu(air);
}

double get_density(AirParcel const& air) {
  return air.get_pa() * get_mu(air) /
         (Constants::Rgas * air.get_temp() * get_gas_mols(air));
}

void to_mmr(AirParcel& air) {
  double mmr = 0.;

  for (int i = 0; i < air.ngas() + air.ncloud(); ++i)
    mmr += air.mgas(i) * air.gas(i)->mu();

  for (int i = 0; i < air.ngas() + air.ncloud(); ++i)
    air.mgas(i) = air.mgas(i) * air.gas(i)->mu() / mmr;
}

void to_vmr(AirParcel& air) {
  double vmr = 0.;

  for (int i = 0; i < air.ngas() + air.ncloud(); ++i)
    vmr += air.mgas(i) / air.gas(i)->mu();

  for (int i = 0; i < air.ngas() + air.ncloud(); ++i)
    air.mgas(i) = air.mgas(i) / air.gas(i)->mu() / vmr;
}

void equilibrate_tp(AirParcel& air, ReactionList const& rlist, double temp,
                    double pres) {
  int iter = 0;
  bool converge = false;

  air.set_tp(temp, pres);

  while (iter < MAX_ITER && !converge) {
    converge = true;
    for (auto r : rlist) converge = converge && r->equilibrate(air);
    iter++;
  }

  if (iter == MAX_ITER) {
    std::cerr << air << std::endl;
    for (auto r : rlist) std::cerr << r->equilibrate(air) << " ";
    std::cerr << std::endl;
    throw "equilibrate_tp fails to converge";
  }
}

void equilibrate_sp(AirParcel& air, ReactionList const& rlist, double entropy,
                    double pres) {
  double Tc, dt = 1.5, Tmin = air.get_temp(), Tmax = air.get_temp();

  equilibrate_tp(air, rlist, Tmin, pres);
  while (get_entropy(air) > entropy) {
    Tmin /= dt;
    equilibrate_tp(air, rlist, Tmin, pres);
  }

  equilibrate_tp(air, rlist, Tmax, pres);
  while (get_entropy(air) < entropy) {
    Tmax *= dt;
    equilibrate_tp(air, rlist, Tmax, pres);
  }

  int error = _root(Tmin, Tmax, PRECISION, &Tc,
                    [&air, &rlist, entropy, pres](double T) {
                      equilibrate_tp(air, rlist, T, pres);
                      return get_entropy(air) - entropy;
                    });

  if (error) {
    std::cerr << air;
    std::cerr << "Tmin = " << Tmin << std::endl;
    std::cerr << "Tmax = " << Tmax << std::endl;
    std::cerr << "Required Entropy = " << entropy << std::endl;
    equilibrate_tp(air, rlist, Tmin, pres);
    std::cerr << "Entropy at Tmin = " << get_entropy(air) << std::endl;
    equilibrate_tp(air, rlist, Tmax, pres);
    std::cerr << "Entropy at Tmax = " << get_entropy(air) << std::endl;
    throw "equilibrate_sp fails to converge";
  }

  double s1, s2;
  AirParcel air1 = air, air2 = air;
  equilibrate_tp(air1, rlist, Tc - 1.E-4, pres);
  s1 = get_entropy(air1);
  equilibrate_tp(air2, rlist, Tc + 1.E-4, pres);
  s2 = get_entropy(air2);

  double xfrac = (s2 - entropy) / (s2 - s1);

  for (int i = 0; i < air.ngas() + air.ncloud(); ++i)
    air.mgas(i) = xfrac * air1.mgas(i) + (1. - xfrac) * air2.mgas(i);

  air.set_tp(Tc, pres);
}

void equilibrate_uv(AirParcel& air, ReactionList const& rlist) {
  double T0 = air.get_temp(), p0 = air.get_pa(), u0 = get_enthalpy(air) + p0,
         Tmin = T0 - 20., Tmax = T0 + 20., Tc;

  int error =
      _root(Tmin, Tmax, PRECISION, &Tc, [&air, &rlist, p0, T0, u0](double T) {
        equilibrate_tp(air, rlist, T, p0 * T / T0);
        return get_enthalpy(air) - u0 - p0 * T / T0;
      });

  assert(!error);

  equilibrate_tp(air, rlist, Tc, p0 * Tc / T0);
}

void precipitate(AirParcel& air, AirParcel& rain) {
  double tmol = get_gas_mols(air);

  for (int i = 0; i < air.ngas(); ++i) air.mgas(i) /= tmol;

  for (int i = 0; i < air.ncloud(); ++i) {
    rain.mcloud(i) += air.mcloud(i);
    air.mcloud(i) = 0.;
  }
}

double KernelModel(double x, double x0[], double s0[], int n,
                   KernelFunction Kernel, void* opts) {
  double result =
      s0[0] * (1. - Kernel(x - x0[0], opts)) + Kernel(x - x0[n - 1], opts);
  for (int i = 1; i < n; ++i)
    result += s0[i] * (Kernel(x - x0[i - 1], opts) - Kernel(x - x0[i], opts));
  return result;
}

#undef PRECISION
#undef MAX_ITER
