// C/C++
#include <stdexcept>

// hapr
#include "toon_mckay89.hpp"

torch::Tensor RadiationBand::RTSolverToon::toonShortwaveSolver(
    torch::Tensor F0_in, torch::Tensor mu_in, torch::Tensor tau_in,
    torch::Tensor w_in, torch::Tensor g_in, torch::Tensor w_surf_in) {
  int nlev = tau_in.size(-1);
  int ncol = tau_in.size(1);

  // Input validation
  if (mu_in.size(0) != ncol || w_in.size(-1) != nlay ||
      g_in.size(-1) != nlay {
    throw std::invalid_argument("Input vectors have incorrect sizes.");
  }

  // Initialize output flux arrays
  auto out = torch::zeros({ncol, nlev, 2}, tau_in.options());
  flx_down = out.select(-1, 0);
  flx_up = out.select(-1, 1);

  // Constants
  const double sqrt3 = std::sqrt(3.0);
  const double sqrt3d2 = sqrt3 / 2.0;
  const double bsurf = 0.0;
  const double btop = 0.0;

  // Check if all single scattering albedos are effectively zero
  bool all_w0_zero = (w_in.array() <= 1.0e-12).all();

  if (all_w0_zero) {  // no scattering
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
    return out;
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

  return out;
}
