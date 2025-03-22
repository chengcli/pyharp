// harp
#include "rtsolver.hpp"

namespace harp {

ToonMcKay89Impl::ToonMcKay89Impl(ToonMcKay89Options const& options)
    : options(options) {
  reset();
}

void ToonMcKay89Impl::reset() {
  // No parameters to initialize
}

torch::Tensor ToonMcKay89Impl::forward(torch::Tensor prop,
                                       std::map<std::string, torch::Tensor>* bc,
                                       torch::optional<torch::Tensor> temf) {
  // Determine flux for each spectral bin using Toon method
  int nlay = ds_.nlyr;
  Eigen::VectorXd w0(nlay), g(nlay);
  Eigen::VectorXd tau_cum(nlay + 1);
  tau_cum.setZero();

  for (int i = 0; i < nlay; ++i) {
    tau_cum(i + 1) = tau_cum(i) + ds_.dtauc[i];
    w0(i) = ds_.ssalb[i];
    g(i) = ds_.pmom[i * (ds_.nmom_nstr + 1) + 1];
  }

  Eigen::VectorXd flux_up(nlay), flux_down(nlay);
  double mu0 = ds_.bc.umu0;
  double surface_albedo = ds_.bc.albedo;
  double Finc = ds_.bc.fbeam;

  // Call shortwave or longwave solver based on the type of radiation
  if (pmy_band_->TestFlag(RadiationFlags::StellarBeam)) {
    Eigen::VectorXd mu0_in(nlay + 1);
    mu0_in.setConstant(mu0);
    toonShortwaveSolver(nlay, Finc, mu0_in, tau_cum, w0, g, surface_albedo,
                        flux_up, flux_down);

  } else if (pmy_band_->TestFlag(RadiationFlags::ThermalEmission)) {
    Eigen::VectorXd temp(nlay + 1);
    Eigen::VectorXd be(nlay + 1);
    for (int i = 0; i < nlay + 1; ++i) {
      temp(i) = ds_.temper[i];
      be(i) = BB_integrate(ds_.temper[i], spec.wav1, spec.wav2);
    }
    double surface_emissivity = ds_.bc.temis;
    toonLongwaveSolver(nlay, be, tau_cum, w0, g, surface_emissivity, flux_up,
                       flux_down);
  }
}

// Toon 1989 shortwave solver, based on Elsie Lee's implementation, which was
// based on CHIMERA code by Mike Line. Reference: Toon, O.B., 1989, JGR, 94,
// 16287-16301.
void RadiationBand::RTSolverToon::toonShortwaveSolver(
    int nlay, double F0_in, const Eigen::VectorXd& mu_in,
    const Eigen::VectorXd& tau_in, const Eigen::VectorXd& w_in,
    const Eigen::VectorXd& g_in, double w_surf_in, Eigen::VectorXd& flx_up,
    Eigen::VectorXd& flx_down) {
  int nlev = nlay + 1;
  // Input validation
  if (mu_in.size() != static_cast<size_t>(nlev) ||
      tau_in.size() != static_cast<size_t>(nlev) ||
      w_in.size() != static_cast<size_t>(nlay) ||
      g_in.size() != static_cast<size_t>(nlay)) {
    throw std::invalid_argument("Input vectors have incorrect sizes.");
  }

  // Initialize output flux arrays
  flx_down = Eigen::VectorXd::Zero(nlev);
  flx_up = Eigen::VectorXd::Zero(nlev);

  // Constants
  const double sqrt3 = std::sqrt(3.0);
  const double sqrt3d2 = sqrt3 / 2.0;
  const double bsurf = 0.0;
  const double btop = 0.0;

  // Check if all albedos are effectively zero
  bool all_w0_zero = (w_in.array() <= 1.0e-12).all();

  if (all_w0_zero) {
    // Direct beam only
    double mu_top = mu_in(nlev - 1);
    double mu_first = mu_in(0);

    if (mu_top > 0.0) {
      if (std::abs(mu_top - mu_first) < 1e-12) {
        // No zenith correction, use regular method
        flx_down = F0_in * mu_top * (-tau_in.array() / mu_top).exp();
      } else {
        // Zenith angle correction using cumulative transmission
        Eigen::VectorXd cum_trans(nlev);
        cum_trans(0) = tau_in(0) / mu_in(0);
        for (int k = 1; k < nlev; ++k) {
          cum_trans(k) =
              cum_trans(k - 1) + (tau_in(k) - tau_in(k - 1)) / mu_in(k);
        }
        flx_down = F0_in * mu_top * (-cum_trans.array()).exp();
      }
      // Adjust the downward flux at the surface layer for surface albedo
      flx_down(nlev - 1) *= (1.0 - w_surf_in);
    }
    // Upward flux remains zero
    return;
  }

  // Delta Eddington scaling
  Eigen::VectorXd w0 = ((1.0 - g_in.array().square()) * w_in.array()) /
                       (1.0 - w_in.array() * g_in.array().square());
  Eigen::VectorXd dtau =
      (1.0 - w_in.array() * g_in.array().square())
          .cwiseProduct((tau_in.segment(1, nlay) - tau_in.head(nlay)).array())
          .matrix();
  Eigen::VectorXd hg = g_in.array() / (1.0 + g_in.array());

  // Initialize tau_total
  Eigen::VectorXd tau_total(nlev);
  tau_total(0) = 0.0;
  for (int k = 0; k < nlay; ++k) {
    tau_total(k + 1) = tau_total(k) + dtau(k);
  }

  // Compute g1, g2, g3, g4
  Eigen::VectorXd g1 = sqrt3d2 * (2.0 - w0.array() * (1.0 + hg.array()));
  Eigen::VectorXd g2 = (sqrt3d2 * w0.array()) * (1.0 - hg.array());
  // Prevent division by zero
  for (int i = 0; i < nlay; ++i) {
    if (std::abs(g2(i)) < 1.0e-10) {
      g2(i) = 1.0e-10;
    }
  }
  // Compute mu_zm at midpoints
  Eigen::VectorXd mu_zm(nlay);
  mu_zm = (mu_in.head(nlay) + mu_in.tail(nlay)) / 2.0;
  Eigen::VectorXd g3 = (1.0 - sqrt3 * hg.array() * mu_zm.array()) / 2.0;
  Eigen::VectorXd g4 = 1.0 - g3.array();

  // Compute lam and gam
  Eigen::VectorXd lam = (g1.array().square() - g2.array().square()).sqrt();
  Eigen::VectorXd gam = (g1.array() - lam.array()) / g2.array();

  // Compute denom and handle denom == 0
  Eigen::VectorXd denom =
      lam.array().square() - (1.0 / (mu_in(nlev - 1) * mu_in(nlev - 1)));
  for (int i = 0; i < nlay; ++i) {
    if (std::abs(denom(i)) < 1e-10) {
      denom(i) = 1.0e-10;
    }
  }

  // Compute Am and Ap
  Eigen::VectorXd Am = F0_in * w0.array() *
                       (g4.array() * (g1.array() + 1.0 / mu_in(nlev - 1)) +
                        g2.array() * g3.array()) /
                       denom.array();
  Eigen::VectorXd Ap = F0_in * w0.array() *
                       (g3.array() * (g1.array() - 1.0 / mu_in(nlev - 1)) +
                        g2.array() * g4.array()) /
                       denom.array();

  // Compute Cpm1 and Cmm1 at the top of the layer
  Eigen::VectorXd Cpm1 =
      Ap.array() * (-tau_total.head(nlay).array() / mu_in(nlev - 1)).exp();
  Eigen::VectorXd Cmm1 =
      Am.array() * (-tau_total.head(nlay).array() / mu_in(nlev - 1)).exp();

  // Compute Cp and Cm at the bottom of the layer
  Eigen::VectorXd Cp =
      Ap.array() *
      (-tau_total.segment(1, nlay).array() / mu_in(nlev - 1)).exp();
  Eigen::VectorXd Cm =
      Am.array() *
      (-tau_total.segment(1, nlay).array() / mu_in(nlev - 1)).exp();

  // Compute exponential terms, clamped to prevent overflow
  Eigen::VectorXd exptrm = (lam.array() * dtau.array()).min(35.0);
  Eigen::VectorXd Ep = exptrm.array().exp();
  Eigen::VectorXd Em = 1.0 / Ep.array();
  Eigen::VectorXd E1 = (Ep.array() + gam.array() * Em.array()).matrix();
  Eigen::VectorXd E2 = (Ep.array() - gam.array() * Em.array()).matrix();
  Eigen::VectorXd E3 = (gam.array() * Ep.array() + Em.array()).matrix();
  Eigen::VectorXd E4 = (gam.array() * Ep.array() - Em.array()).matrix();

  // Initialize Af, Bf, Cf, Df
  int l = 2 * nlay;
  Eigen::VectorXd Af_vec = Eigen::VectorXd::Zero(l);
  Eigen::VectorXd Bf_vec = Eigen::VectorXd::Zero(l);
  Eigen::VectorXd Cf_vec = Eigen::VectorXd::Zero(l);
  Eigen::VectorXd Df_vec = Eigen::VectorXd::Zero(l);

  // Boundary conditions at the top
  Af_vec(0) = 0.0;
  Bf_vec(0) = gam(0) + 1.0;
  Cf_vec(0) = gam(0) - 1.0;
  Df_vec(0) = btop - Cmm1(0);
  for (int i = 1, n = 1; i < l - 1; i += 2, ++n) {
    if (n >= nlay) {
      throw std::out_of_range(
          "Index out of range in sw_Toon89 Af, Bf, Cf, Df population.");
    }
    Af_vec(i) = (E1(n - 1) + E3(n - 1)) * (gam(n) - 1.0);
    Bf_vec(i) = (E2(n - 1) + E4(n - 1)) * (gam(n) - 1.0);
    Cf_vec(i) = 2.0 * (1.0 - gam(n) * gam(n));
    Df_vec(i) = (gam(n) - 1.0) * (Cpm1(n) - Cp(n - 1)) +
                (1.0 - gam(n)) * (Cm(n - 1) - Cmm1(n));
  }

  // Populate Af, Bf, Cf, Df for even indices
  // Start from n=1 to avoid negative indexing (Cp(n-1) when n=0)
  for (int i = 2, n = 1; i < l - 1; i += 2, ++n) {
    if (n >= nlay) {
      throw std::out_of_range(
          "Index out of range in sw_Toon89 Af, Bf, Cf, Df population.");
    }
    Af_vec(i) = 2.0 * (1.0 - gam(n) * gam(n));
    Bf_vec(i) = (E1(n - 1) - E3(n - 1)) * (1.0 + gam(n));
    Cf_vec(i) = (E1(n - 1) + E3(n - 1)) * (gam(n) - 1.0);
    Df_vec(i) =
        E3(n - 1) * (Cpm1(n) - Cp(n - 1)) + E1(n - 1) * (Cm(n - 1) - Cmm1(n));
  }

  // Boundary conditions at l (last index)
  Af_vec(l - 1) = E1(nlay - 1) - w_surf_in * E3(nlay - 1);
  Bf_vec(l - 1) = E2(nlay - 1) - w_surf_in * E4(nlay - 1);
  Cf_vec(l - 1) = 0.0;
  Df_vec(l - 1) = bsurf - Cp(nlay - 1) + w_surf_in * Cm(nlay - 1);

  // Prepare a, b, c, d for the solver
  Eigen::VectorXd a_tridiag = Af_vec.segment(1, l - 1);
  Eigen::VectorXd b_tridiag = Bf_vec;
  Eigen::VectorXd c_tridiag = Cf_vec.segment(0, l - 1);
  Eigen::VectorXd d_tridiag = Df_vec;

  // Solve the tridiagonal system
  Eigen::VectorXd xk =
      tridiagonal_solver(a_tridiag, b_tridiag, c_tridiag, d_tridiag);

  // Compute xk1 and xk2 from xk
  Eigen::VectorXd xk1(nlay);
  Eigen::VectorXd xk2(nlay);
  for (int idx = 0; idx < nlay; ++idx) {
    int two_n = 2 * idx;
    if (two_n + 1 >= xk.size()) {
      throw std::out_of_range("Index out of range when accessing xk.");
    }
    xk1(idx) = xk(two_n) + xk(two_n + 1);
    xk2(idx) = xk(two_n) - xk(two_n + 1);
    if (std::abs(xk2(idx) / xk(two_n)) < 1e-30) {
      xk2(idx) = 0.0;
    }
  }

  // Populate flx_up and flx_down for layers 1 to nlay
  flx_up.head(nlay) =
      (xk1.array() + gam.array() * xk2.array() + Cpm1.array()).matrix();
  flx_down.head(nlay) =
      (xk1.array() * gam.array() + xk2.array() + Cmm1.array()).matrix();

  // Compute flx_up and flx_down at level nlev
  flx_up(nlev - 1) = xk1(nlay - 1) * std::exp(1.0) +
                     gam(nlay - 1) * xk2(nlay - 1) * std::exp(-1.0) +
                     Cp(nlay - 1);
  flx_down(nlev - 1) = xk1(nlay - 1) * std::exp(1.0) * gam(nlay - 1) +
                       xk2(nlay - 1) * std::exp(-1.0) + Cm(nlay - 1);

  // Compute dir flux
  Eigen::VectorXd dir = Eigen::VectorXd::Zero(nlev);
  double mu_top_nonzero = mu_in(nlev - 1);
  double mu_first_nonzero = mu_in(0);
  if (std::abs(mu_top_nonzero - mu_first_nonzero) < 1e-12) {
    // No zenith correction
    dir = F0_in * mu_top_nonzero * (-tau_in.array() / mu_top_nonzero).exp();
  } else {
    // Zenith angle correction
    Eigen::VectorXd cum_trans(nlev);
    cum_trans(0) = tau_total(0) / mu_in(0);
    for (int k = 1; k < nlev; ++k) {
      cum_trans(k) =
          cum_trans(k - 1) + (tau_total(k) - tau_total(k - 1)) / mu_in(k);
    }
    dir = F0_in * mu_top_nonzero * (-cum_trans.array()).exp();
  }

  // Adjust the downward flux at the surface layer for surface albedo
  dir(nlev - 1) *= (1.0 - w_surf_in);

  // for(int i=0; i <nlev; ++i) std::cout << "flux_up: " << flx_up(i) << "
  // flux_down: " << flx_down(i) << " dirflux_down: " << dir(i) << std::endl;
  //  Add the direct beam contribution
  flx_down += dir;

  // Ensure no negative fluxes due to numerical errors
  flx_down = flx_down.cwiseMax(0.0);
  flx_up = flx_up.cwiseMax(0.0);
}

// Toon 1989 longwave solver, based on Elsie Lee's implementation in
// Exo-FMS_column_ck, which was based on CHIMERA code by Mike Line. Reference:
// Toon, O.B., 1989, JGR, 94, 16287-16301.
void RadiationBand::RTSolverToon::toonLongwaveSolver(
    int nlay, const Eigen::VectorXd& be, const Eigen::VectorXd& tau_in,
    const Eigen::VectorXd& w_in, const Eigen::VectorXd& g_in, double a_surf_in,
    Eigen::VectorXd& flx_up, Eigen::VectorXd& flx_down) {
  // Constants for Gaussian Quadrature
  static const int nmu = 2;
  static const Eigen::VectorXd uarr =
      (Eigen::VectorXd(nmu) << 0.21132487, 0.78867513).finished();
  static const Eigen::VectorXd w =
      (Eigen::VectorXd(nmu) << 0.5, 0.5).finished();
  static const Eigen::VectorXd wuarr = uarr.array() * w.array();
  static const double ubari = 0.5;
  static const double twopi = 6.283185307179586;

  int nlev = nlay + 1;
  // Define l, lm2, lm1
  int l = 2 * nlay;
  int lm2 = l - 2;
  int lm1 = l - 1;

  // Initialize work variables
  Eigen::VectorXd tau(nlev);
  Eigen::VectorXd dtau_in = tau_in.segment(1, nlay) - tau_in.head(nlay);
  Eigen::VectorXd dtau =
      (1.0 - w_in.array() * g_in.array().square()) * dtau_in.array();
  Eigen::VectorXd w0 = ((1.0 - g_in.array().square()) * w_in.array()) /
                       (1.0 - w_in.array() * g_in.array().square());
  Eigen::VectorXd hg = g_in.array() / (1.0 + g_in.array());

  // Initialize cumulative optical depth
  tau(0) = 0.0;
  for (int k = 0; k < nlay; ++k) {
    tau(k + 1) = tau(k) + dtau(k);
  }

  // Compute alp, lam, gam, term
  Eigen::VectorXd alp =
      ((1.0 - w0.array()) / (1.0 - w0.array() * hg.array())).sqrt();
  Eigen::VectorXd lam = (alp.array() * (1.0 - w0.array() * hg.array())) / ubari;
  Eigen::VectorXd gam = (1.0 - alp.array()) / (1.0 + alp.array());
  Eigen::VectorXd term = ubari / (1.0 - w0.array() * hg.array());

  // Compute B0 and B1
  Eigen::VectorXd B0 = Eigen::VectorXd::Zero(nlay);
  Eigen::VectorXd B1 = Eigen::VectorXd::Zero(nlay);
  for (int k = 0; k < nlay; ++k) {
    if (dtau(k) <= 1.0e-6) {
      // For low optical depths use the isothermal approximation
      B1(k) = 0.0;
      B0(k) = 0.5 * (be(k + 1) + be(k));
    } else {
      B1(k) = (be(k + 1) - be(k)) / dtau(k);
      B0(k) = be(k);
    }
  }

  // Compute Cpm1, Cmm1, Cp, Cm
  Eigen::VectorXd Cpm1 = B0.array() + B1.array() * term.array();
  Eigen::VectorXd Cmm1 = B0.array() - B1.array() * term.array();
  Eigen::VectorXd Cp =
      B0.array() + B1.array() * dtau.array() + B1.array() * term.array();
  Eigen::VectorXd Cm =
      B0.array() + B1.array() * dtau.array() - B1.array() * term.array();

  // Compute tautop, Btop, Bsurf, bottom
  double tautop = dtau(0) * std::exp(-1.0);
  double Btop = (1.0 - std::exp(-tautop / ubari)) * be(0);
  double Bsurf = be(nlev - 1);
  double bottom = Bsurf + B1(nlay - 1) * ubari;

  // Compute exponential terms
  Eigen::VectorXd exptrm = (lam.array() * dtau.array()).min(35.0);
  Eigen::VectorXd Ep = exp(exptrm.array());
  Eigen::VectorXd Em = 1.0 / Ep.array();

  // Compute E1, E2, E3, E4
  Eigen::VectorXd E1 = Ep.array() + gam.array() * Em.array();
  Eigen::VectorXd E2 = Ep.array() - gam.array() * Em.array();
  Eigen::VectorXd E3 = gam.array() * Ep.array() + Em.array();
  Eigen::VectorXd E4 = gam.array() * Ep.array() - Em.array();

  // Initialize Af, Bf, Cf, Df
  Eigen::VectorXd Af_vec = Eigen::VectorXd::Zero(l);
  Eigen::VectorXd Bf_vec = Eigen::VectorXd::Zero(l);
  Eigen::VectorXd Cf_vec = Eigen::VectorXd::Zero(l);
  Eigen::VectorXd Df_vec = Eigen::VectorXd::Zero(l);

  // Boundary conditions at the top
  Af_vec(0) = 0.0;
  Bf_vec(0) = gam(0) + 1.0;
  Cf_vec(0) = gam(0) - 1.0;
  Df_vec(0) = Btop - Cmm1(0);

  // Populate Af, Bf, Cf, Df for odd indices (i=1,3,...)
  // Initialize n to 1 as per user suggestion
  int n = 1;
  for (int i = 1; i < l - 1; i += 2, ++n) {
    if (n > nlay) {
      throw std::out_of_range(
          "Index out of range in lw_Toon89 Af, Bf, Cf, Df population.");
    }
    Af_vec(i) = (E1(n - 1) + E3(n - 1)) * (gam(n) - 1.0);
    Bf_vec(i) = (E2(n - 1) + E4(n - 1)) * (gam(n) - 1.0);
    Cf_vec(i) = 2.0 * (1.0 - gam(n) * gam(n));
    Df_vec(i) = (gam(n) - 1.0) * (Cpm1(n) - Cp(n - 1)) +
                (1.0 - gam(n)) * (Cm(n - 1) - Cmm1(n));
  }

  // Populate Af, Bf, Cf, Df for even indices (i=2,4,...)
  n = 1;  // Reset n to 1 for even indices
  for (int i = 2; i < l - 1; i += 2, ++n) {
    if (n > nlay) {
      throw std::out_of_range(
          "Index out of range in lw_Toon89 Af, Bf, Cf, Df population.");
    }
    Af_vec(i) = 2.0 * (1.0 - gam(n - 1) * gam(n - 1));
    Bf_vec(i) = (E1(n - 1) - E3(n - 1)) * (1.0 + gam(n));
    Cf_vec(i) = (E1(n - 1) + E3(n - 1)) * (gam(n) - 1.0);
    Df_vec(i) =
        E3(n - 1) * (Cpm1(n) - Cp(n - 1)) + E1(n - 1) * (Cm(n - 1) - Cmm1(n));
  }

  // Boundary conditions at the last index
  Af_vec(l - 1) = E1(nlay - 1) - a_surf_in * E3(nlay - 1);
  Bf_vec(l - 1) = E2(nlay - 1) - a_surf_in * E4(nlay - 1);
  Cf_vec(l - 1) = 0.0;
  Df_vec(l - 1) = Bsurf - Cp(nlay - 1) + a_surf_in * Cm(nlay - 1);

  // Prepare a, b, c, d for the tridiagonal solver
  Eigen::VectorXd a_tridiag = Af_vec.segment(1, l - 1);
  Eigen::VectorXd b_tridiag = Bf_vec;
  Eigen::VectorXd c_tridiag = Cf_vec.segment(0, l - 1);
  Eigen::VectorXd d_tridiag = Df_vec;

  // Solve the tridiagonal system
  Eigen::VectorXd xkk =
      tridiagonal_solver(a_tridiag, b_tridiag, c_tridiag, d_tridiag);

  // Compute xk1 and xk2 from xkk
  Eigen::VectorXd xk1 = Eigen::VectorXd::Zero(nlay);
  Eigen::VectorXd xk2 = Eigen::VectorXd::Zero(nlay);
  for (int idx = 0; idx < nlay; ++idx) {
    int two_n = 2 * idx;
    if (two_n + 1 >= xkk.size()) {
      throw std::out_of_range("Index out of range when accessing xk.");
    }
    xk1(idx) = xkk(two_n) + xkk(two_n + 1);
    xk2(idx) = xkk(two_n) - xkk(two_n + 1);
    if (std::abs(xk2(idx) / xkk(two_n)) < 1e-30) {
      xk2(idx) = 0.0;
    }
  }

  // Apply conditional "where (w0 <= 1e-4)"
  // Using Eigen's select function for element-wise conditional assignments
  Eigen::ArrayXd mask = (w0.array() <= 1e-4).cast<double>();

  Eigen::VectorXd g_var_final =
      mask * 0.0 +
      (1.0 - mask) * (twopi * w0.array() * xk1.array() *
                      (1.0 + hg.array() * alp.array()) / (1.0 + alp.array()));
  Eigen::VectorXd h_var_final =
      mask * 0.0 +
      (1.0 - mask) * (twopi * w0.array() * xk2.array() *
                      (1.0 - hg.array() * alp.array()) / (1.0 + alp.array()));
  Eigen::VectorXd xj_final =
      mask * 0.0 +
      (1.0 - mask) * (twopi * w0.array() * xk1.array() *
                      (1.0 - hg.array() * alp.array()) / (1.0 + alp.array()));
  Eigen::VectorXd xk_final =
      mask * 0.0 +
      (1.0 - mask) * (twopi * w0.array() * xk2.array() *
                      (1.0 + hg.array() * alp.array()) / (1.0 + alp.array()));
  Eigen::VectorXd alpha1_vec_final =
      mask * (twopi * B0.array()) +
      (1.0 - mask) * (twopi * (B0.array() +
                               B1.array() * (ubari * w0.array() * hg.array() /
                                             (1.0 - w0.array() * hg.array()))));
  Eigen::VectorXd alpha2_vec_final = twopi * B1.array();
  Eigen::VectorXd sigma1_final =
      mask * (twopi * B0.array()) +
      (1.0 - mask) * (twopi * (B0.array() -
                               B1.array() * (ubari * w0.array() * hg.array() /
                                             (1.0 - w0.array() * hg.array()))));
  Eigen::VectorXd sigma2_final = alpha2_vec_final;

  // Compute obj, epp, em1, obj2, epp2
  Eigen::VectorXd obj_vec = (lam.array() * dtau.array()).min(35.0);
  Eigen::VectorXd epp_vec_final = obj_vec.array().exp();
  Eigen::VectorXd em1_final = 1.0 / epp_vec_final.array();
  Eigen::VectorXd obj2_vec = (0.5 * lam.array() * dtau.array()).min(35.0);
  Eigen::VectorXd epp2_vec_final = obj2_vec.array().exp();

  // Initialize and resize flux arrays to ensure correct dimensions
  flx_up = Eigen::VectorXd::Zero(nlev);
  flx_down = Eigen::VectorXd::Zero(nlev);

  // Initialize lw_up_g and lw_down_g
  Eigen::VectorXd lw_up_g = Eigen::VectorXd::Zero(nlev);
  Eigen::VectorXd lw_down_g = Eigen::VectorXd::Zero(nlev);

  // Loop over m=1 to nmu (0 to nmu-1 in C++)
  for (int m = 0; m < nmu; ++m) {
    // Compute em2 and em3
    Eigen::VectorXd em2_vec = (-dtau.array() / uarr(m)).exp();
    Eigen::VectorXd em3_vec = em1_final.array() * em2_vec.array();

    // Downward loop
    lw_down_g(0) = twopi * (1.0 - std::exp(-tautop / uarr(m))) * be(0);
    for (int k = 0; k < nlay; ++k) {
      lw_down_g(k + 1) =
          lw_down_g(k) * em2_vec(k) +
          (xj_final(k) / (lam(k) * uarr(m) + 1.0)) *
              (epp_vec_final(k) - em2_vec(k)) +
          (xk_final(k) / (lam(k) * uarr(m) - 1.0)) *
              (em2_vec(k) - em1_final(k)) +
          sigma1_final(k) * (1.0 - em2_vec(k)) +
          sigma2_final(k) * (uarr(m) * em2_vec(k) + dtau(k) - uarr(m));
    }

    // Upward loop
    lw_up_g(nlev - 1) = twopi * (Bsurf + B1(nlay - 1) * uarr(m));
    for (int k = nlay - 1; k >= 0; --k) {
      lw_up_g(k) =
          lw_up_g(k + 1) * em2_vec(k) +
          (g_var_final(k) / (lam(k) * uarr(m) - 1.0)) *
              (epp_vec_final(k) * em2_vec(k) - 1.0) +
          (h_var_final(k) / (lam(k) * uarr(m) + 1.0)) * (1.0 - em3_vec(k)) +
          alpha1_vec_final(k) * (1.0 - em2_vec(k)) +
          alpha2_vec_final(k) * (uarr(m) - (dtau(k) + uarr(m)) * em2_vec(k));
    }

    // Sum up flux arrays with Gaussian quadrature weights
    flx_down += (lw_down_g.array() * wuarr(m)).matrix();
    flx_up += (lw_up_g.array() * wuarr(m)).matrix();
  }
  // for(int i=0; i <nlev; ++i) std::cout << "flux_up: " << flx_up(i) << "
  // flux_down: " << flx_down(i) << std::endl;
}

}  // namespace harp
