// RT solvers based on Toon 1989 method by Xi Zhang
// Reference: Toon, O.B., 1989, JGR, 94, 16287-16301.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>

// external
#include <yaml-cpp/yaml.h>

// athena
#include <athena/coordinates/coordinates.hpp>
#include <athena/mesh/mesh.hpp>

// application
#include <application/application.hpp>
#include <application/exceptions.hpp>

// climath
#include <climath/interpolation.h>

// canoe
#include <constants.hpp>
#include <impl.hpp>

// astro
#include <astro/celestrial_body.hpp>

// exo3
#include <exo3/cubed_sphere.hpp>
#include <exo3/cubed_sphere_utility.hpp>

// harp
#include "radiation.hpp"
#include "rt_solvers.hpp"

#ifdef RT_DISORT

RadiationBand::RTSolverToon::RTSolverToon(RadiationBand *pmy_band,
                                          YAML::Node const &rad)
    : RTSolver(pmy_band, "Toon") {
  Application::Logger app("harp");
  app->Log("Toon solver initialized for band " + pmy_band_->GetName());
}

//! \todo update based on band outdir
void RadiationBand::RTSolverToon::Resize(int nlyr, int nstr) {
  RadiationBand::RTSolver::Resize(nlyr, nstr);
  Unseal();
  SetAtmosphereDimension(nlyr, nstr, nstr);
  Seal();
}

void RadiationBand::RTSolverToon::Prepare(MeshBlock const *pmb, int k, int j) {
  auto &wmin = pmy_band_->wrange_.first;
  auto &wmax = pmy_band_->wrange_.second;

  Real dist_au = 1.0;
  Direction ray = pmb->pimpl->prad->GetRayInput(0);
  auto planet = pmb->pimpl->planet;

  if (planet && pmy_band_->TestFlag(RadiationFlags::TimeDependent)) {
    Real time = pmb->pmy_mesh->time;
    Real lat, lon;

    CubedSphereUtility::get_latlon_on_sphere(&lat, &lon, pmb, k, j, pmb->is);

    ray = planet->ParentZenithAngle(time, lat, lon);
    dist_au = planet->ParentDistanceInAu(time);
  } else {
    if (pmy_band_->HasPar("umu0")) {
      ray.mu = pmy_band_->GetPar<Real>("umu0");
    }

    if (pmy_band_->HasPar("phi0")) {
      ray.phi = pmy_band_->GetPar<Real>("phi0");
    }

    if (pmy_band_->HasPar("dist_au")) {
      dist_au = pmy_band_->GetPar<Real>("dist_au");
    }
  }

  // pack temperature
  if (pmy_band_->TestFlag(RadiationFlags::ThermalEmission)) {
    pmy_band_->packTemperature();
  }

  // pack spectral properties
  pmy_band_->packSpectralProperties();
  ds_.bc.umu0 = ray.mu > 1.E-3 ? ray.mu : 1.E-3;

  if (pmy_band_->TestFlag(RadiationFlags::BroadBand)) {
    // stellar source function overrides fbeam
    if (pmy_band_->HasPar("S0")) {
      ds_.bc.fbeam = pmy_band_->GetPar<Real>("S0");
    } else if (pmy_band_->HasPar("temp0")) {
      Real temp0 = pmy_band_->GetPar<Real>("temp0");
      ds_.bc.fbeam = Constants::stefanBoltzmann * pow(temp0, 4);
    } else if (planet && planet->HasParentFlux()) {
      ds_.bc.fbeam = planet->ParentInsolationFlux(wmin, wmax, 1.);
    } else {
      ds_.bc.fbeam = 0.;
    }
    ds_.bc.fbeam /= dist_au * dist_au;
  }

  pmb->pcoord->Face1Area(k, j, pmb->is, pmb->ie + 1, farea_);
  pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol_);
}

void RadiationBand::RTSolverToon::CalBandFlux(MeshBlock const *pmb, int k,
                                              int j) {
  Real dist_au = 1.0;
  auto planet = pmb->pimpl->planet;

  if (planet && pmy_band_->TestFlag(RadiationFlags::TimeDependent)) {
    dist_au = planet->ParentDistanceInAu(pmb->pmy_mesh->time);
  } else if (pmy_band_->HasPar("dist_au")) {
    dist_au = pmy_band_->GetPar<Real>("dist_au");
  }

  // loop over spectral grids in the band
  bool override_with_stellar_spectra = false;
  if (!pmy_band_->TestFlag(RadiationFlags::BroadBand) &&
      !pmy_band_->HasPar("S0") && !pmy_band_->HasPar("temp0") && planet &&
      planet->HasParentFlux()) {
    override_with_stellar_spectra = true;
  }

  pmy_band_->pexv->GatherAll(pmb);
  if (pmy_band_->TestFlag(RadiationFlags::ThermalEmission)) {
    pmy_band_->unpackTemperature(&ds_);
  }

  int b = 0;
  for (auto &spec : pmy_band_->pgrid_->spec) {
    if (override_with_stellar_spectra) {
      // stellar source function
      ds_.bc.fbeam =
          planet->ParentInsolationFlux(spec.wav1, spec.wav2, dist_au);
    }

    // Transfer spectral grid data
    pmy_band_->unpackSpectralProperties(b, &ds_);

    // add spectral bin flux
    addToonFlux(pmb->pcoord, b++, k, j, pmb->is, pmb->ie + 1, flux_up,
                flux_down);
  }
}

void RadiationBand::RTSolverToon::addToonFlux(
    Coordinates const *pcoord, int b, int k, int j, int il, int iu,
    const Eigen::VectorXd &flux_up, const Eigen::VectorXd &flux_down) {
  auto &bflxup = pmy_band_->bflxup;
  auto &bflxdn = pmy_band_->bflxdn;

  auto &flxup = pmy_band_->flxup_;
  auto &flxdn = pmy_band_->flxdn_;
  auto const &spec = pmy_band_->pgrid_->spec;

  int rank_in_column = pmy_band_->pexv->GetRankInGroup();

  // Accumulate flux from spectral bins
  for (int i = il; i <= iu; ++i) {
    int m = ds_.nlyr - (rank_in_column * (iu - il) + i - il);
    // Flux up
    flxup(b, k, j, i) = flux_up(m);
    // Flux down
    flxdn(b, k, j, i) = flux_down(m);

    bflxup(k, j, i) += spec[b].wght * flxup(b, k, j, i);
    bflxdn(k, j, i) += spec[b].wght * flxdn(b, k, j, i);
  }

  // Spherical correction
  Real volh;
  Real bflxup_iu = bflxup(k, j, iu);
  Real bflxdn_iu = bflxdn(k, j, iu);

  for (int i = iu - 1; i >= il; --i) {
    // Upward
    volh = (bflxup_iu - bflxup(k, j, i)) / pcoord->dx1f(i) * vol_(i);
    bflxup_iu = bflxup(k, j, i);
    bflxup(k, j, i) = (bflxup(k, j, i + 1) * farea_(i + 1) - volh) / farea_(i);

    // Downward
    volh = (bflxdn_iu - bflxdn(k, j, i)) / pcoord->dx1f(i) * vol_(i);
    bflxdn_iu = bflxdn(k, j, i);
    bflxdn(k, j, i) = (bflxdn(k, j, i + 1) * farea_(i + 1) - volh) / farea_(i);
  }

  /*
  for (int i = iu; i >= il; --i) {
    std::cout << "i: " << iu-i+1 <<" flxup: " << bflxup(k, j, i) << " flxdn: "
  << bflxdn(k, j, i) << " fluxdiff: " << bflxup(k, j, i) - bflxdn(k, j, i) <<
  std::endl;
  }
*/
}

// Inegrate Planck function over a band, based on cdisort
double RadiationBand::RTSolverToon::BB_integrate(double T, double wn1,
                                                 double wn2) {
  if (T < 1e-4 || wn2 < wn1 || wn1 < 0.0) {
    throw std::invalid_argument(
        "BB_integrate: Invalid temperature or wavenumbers");
  }

  constexpr double C2 = 1.438786;       // h * c / k in units cm * K
  constexpr double SIGMA = 5.67032e-8;  // Stefan-Boltzmann constant in W/m²K⁴
  constexpr double VCUT = 1.5;
  constexpr double sigdpi = SIGMA / M_PI;
  const double vmax = std::log(DBL_MAX);
  const double conc = 15.0 / std::pow(M_PI, 4);  // Now computed at runtime
  constexpr double c1 = 1.1911e-18;  // h * c^2, in units W/(m² * sr * cm⁻⁴)
  constexpr double A1 = 1.0 / 3.0;
  constexpr double A2 = -1.0 / 8.0;
  constexpr double A3 = 1.0 / 60.0;
  constexpr double A4 = -1.0 / 5040.0;
  constexpr double A5 = 1.0 / 272160.0;
  constexpr double A6 = -1.0 / 13305600.0;
  // Helper function to compute Planck integrand value
  auto planck_function = [](double v) {
    return std::pow(v, 3) / (std::exp(v) - 1.0);
  };

  // Handle the case where wn1 == wn2
  if (wn1 == wn2) {
    double wn = wn1;
    double arg = std::exp(-C2 * wn / T);
    return c1 * std::pow(wn, 3) * arg / (1.0 - arg);
  }

  double v[2] = {C2 * wn1 / T, C2 * wn2 / T};
  double p[2] = {0.0, 0.0}, d[2] = {0.0, 0.0};
  int smallv = 0;

  // Handle different cases for wavenumbers
  for (int i = 0; i <= 1; ++i) {
    if (v[i] < VCUT) {
      // Use power series expansion
      smallv++;
      double vsq = v[i] * v[i];
      p[i] =
          conc * vsq * v[i] *
          (A1 + v[i] * (A2 + v[i] * (A3 + vsq * (A4 + vsq * (A5 + vsq * A6)))));
    } else {
      // Use exponential series expansion
      int mmax = 1;
      static const double vcp[7] = {10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0};
      while (v[i] < vcp[mmax - 1] && mmax < 7) {
        ++mmax;
      }

      double ex = std::exp(-v[i]);
      double exm = 1.0;
      d[i] = 0.0;

      for (int m = 1; m <= mmax; ++m) {
        double mv = static_cast<double>(m) * v[i];
        exm *= ex;
        d[i] += exm * (6.0 + mv * (6.0 + mv * (3.0 + mv))) / (m * m);
      }
      d[i] *= conc;
    }
  }

  double ans;
  if (smallv == 2) {
    // Both wavenumbers are small
    ans = p[1] - p[0];
  } else if (smallv == 1) {
    // One wavenumber is small, the other is large
    ans = 1.0 - p[0] - d[1];
  } else {
    // Both wavenumbers are large
    ans = d[0] - d[1];
  }

  ans *= sigdpi * T * T * T * T;

  if (ans == 0.0) {
    std::cerr << "BB_integrate: Warning - result is zero; possible underflow"
              << std::endl;
  }

  return ans;
}

// Tridiagonal Solver using the Thomas Algorithm
inline Eigen::VectorXd RadiationBand::RTSolverToon::tridiagonal_solver(
    const Eigen::VectorXd &a, const Eigen::VectorXd &b,
    const Eigen::VectorXd &c, const Eigen::VectorXd &d) {
  int l = b.size();
  if (a.size() != static_cast<size_t>(l - 1) ||
      c.size() != static_cast<size_t>(l - 1) ||
      d.size() != static_cast<size_t>(l)) {
    throw std::invalid_argument(
        "Incorrect vector sizes for tridiagonal_solver.");
  }

  Eigen::VectorXd c_prime(l - 1);
  Eigen::VectorXd d_prime(l);

  // Forward sweep
  c_prime(0) = c(0) / b(0);
  d_prime(0) = d(0) / b(0);
  for (int i = 1; i < l - 1; ++i) {
    double denom = b(i) - a(i - 1) * c_prime(i - 1);
    if (std::abs(denom) < 1e-12) {
      throw std::runtime_error(
          "Tridiagonal solver failed: near-zero denominator.");
    }
    c_prime(i) = c(i) / denom;
    d_prime(i) = (d(i) - a(i - 1) * d_prime(i - 1)) / denom;
  }

  // Last equation
  double denom_last = b(l - 1) - a(l - 2) * c_prime(l - 2);
  if (std::abs(denom_last) < 1e-12) {
    throw std::runtime_error(
        "Tridiagonal solver failed: near-zero denominator at last equation.");
  }
  d_prime(l - 1) = (d(l - 1) - a(l - 2) * d_prime(l - 2)) / denom_last;

  // Back substitution
  Eigen::VectorXd x(l);
  x(l - 1) = d_prime(l - 1);
  for (int i = l - 2; i >= 0; --i) {
    x(i) = d_prime(i) - c_prime(i) * x(i + 1);
  }

  return x;
}

#endif
