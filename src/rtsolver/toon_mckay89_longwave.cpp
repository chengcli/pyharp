
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
