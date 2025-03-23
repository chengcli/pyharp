// C/C++
#include <stdexcept>

// harp
#include <math/tridiag.hpp>

#include "toon_mckay89.hpp"

torch::Tensor ToonMcKay89Impl::longwave_solver(torch::Tensor be,
                                               torch::Tensor tau_in,
                                               torch::Tensor w_in,
                                               torch::Tensor g_in,
                                               torch::Tensor w_surf_in) {
  const int nmu = 2;
  const auto dtype = be.scalar_type();
  const auto device = be.device();
  const auto uarr =
      torch::tensor({0.21132487, 0.78867513},
                    torch::TensorOptions().dtype(dtype).device(device));
  const auto w = torch::tensor(
      {0.5, 0.5}, torch::TensorOptions().dtype(dtype).device(device));
  const auto wuarr = uarr * w;
  const double ubari = 0.5;
  const double twopi = 6.283185307179586;

  int nlev = nlay + 1;

  auto out = torch::zeros({ncol, nlev, 2}, tau_cum.options());
  flx_down = out.select(-1, 0);
  flx_up = out.select(-1, 1);

  // dtau = (1 - w * g^2) * (tau_in[1:] - tau_in[:-1])
  auto dtau_in = tau_in.narrow(-1, 1, nlay) - tau_in.narrow(-1, 0, nlay);
  auto g2 = g_in * g_in;
  auto dtau = (1.0 - w_in * g2) * dtau_in;

  auto w0 = ((1.0 - g2) * w_in) / (1.0 - w_in * g2);
  auto hg = g_in / (1.0 + g_in);

  auto tau0 = torch::zeros_like(tau_in.select(-1, 0).unsqueeze(-1));
  auto tau = torch::cat({tau0, dtau.cumsum(-1)}, -1);

  auto denom = 1.0 - w0 * hg;
  auto alp = ((1.0 - w0) / denom).sqrt();
  auto lam = alp * denom / ubari;
  auto gam = (1.0 - alp) / (1.0 + alp);
  auto term = ubari / denom;

  auto B0 = torch::empty_like(w0);
  auto B1 = torch::empty_like(w0);
  auto small_dtau_mask = dtau <= 1.0e-6;

  auto be_k = be.narrow(-1, 0, nlay);
  auto be_k1 = be.narrow(-1, 1, nlay);

  B1.masked_fill_(small_dtau_mask, 0.0);
  B0.masked_scatter_(small_dtau_mask,
                     0.5 * (be_k + be_k1).masked_select(small_dtau_mask));

  auto B1_alt = (be_k1 - be_k) / dtau;
  B1.masked_scatter_(~small_dtau_mask, B1_alt.masked_select(~small_dtau_mask));
  B0.masked_scatter_(~small_dtau_mask, be_k.masked_select(~small_dtau_mask));

  auto term_B1 = B1 * term;
  auto Cpm1 = B0 + term_B1;
  auto Cmm1 = B0 - term_B1;
  auto dtau_B1 = B1 * dtau;
  auto Cp = B0 + dtau_B1 + term_B1;
  auto Cm = B0 + dtau_B1 - term_B1;

  auto tautop = dtau.select(-1, 0) * std::exp(-1.0);
  auto Btop = (1.0 - (tautop / ubari).neg().exp()) * be.select(-1, 0);
  auto Bsurf = be.select(-1, nlev - 1);
  auto bottom = Bsurf + B1.select(-1, nlay - 1) * ubari;

  auto exptrm = torch::min(lam * dtau, torch::tensor(35.0, lam.options()));
  auto Ep = exptrm.exp();
  auto Em = 1.0 / Ep;

  auto E1 = Ep + gam * Em;
  auto E2 = Ep - gam * Em;
  auto E3 = gam * Ep + Em;
  auto E4 = gam * Ep - Em;

  // ========================== Fill Af, Bf, Cf, Df ==========================
  int l = 2 * nlay;
  torch::Tensor Af_vec = torch::zeros_like(torch::empty({l}, dtau.options()))
                             .expand_as(B0)
                             .unsqueeze(-1)
                             .repeat({1, 1, l});
  torch::Tensor Bf_vec = Af_vec.clone();
  torch::Tensor Cf_vec = Af_vec.clone();
  torch::Tensor Df_vec = Af_vec.clone();

  Bf_vec.select(-1, 0).copy_(gam.select(-1, 0) + 1.0);
  Cf_vec.select(-1, 0).copy_(gam.select(-1, 0) - 1.0);
  Df_vec.select(-1, 0).copy_(Btop.unsqueeze(-1) - Cmm1.select(-1, 0));

  for (int i = 1, n = 1; i < l - 1; i += 2, ++n) {
    auto gam_n = gam.select(-1, n);
    auto gam_nm1 = gam.select(-1, n - 1);

    auto E1_n = E1.select(-1, n - 1);
    auto E2_n = E2.select(-1, n - 1);
    auto E3_n = E3.select(-1, n - 1);
    auto E4_n = E4.select(-1, n - 1);

    auto Cp_nm1 = Cp.select(-1, n - 1);
    auto Cpm1_n = Cpm1.select(-1, n);
    auto Cm_nm1 = Cm.select(-1, n - 1);
    auto Cmm1_n = Cmm1.select(-1, n);

    Af_vec.select(-1, i).copy_((E1_n + E3_n) * (gam_n - 1.0));
    Bf_vec.select(-1, i).copy_((E2_n + E4_n) * (gam_n - 1.0));
    Cf_vec.select(-1, i).copy_(2.0 * (1.0 - gam_n * gam_n));
    Df_vec.select(-1, i).copy_((gam_n - 1.0) * (Cpm1_n - Cp_nm1) +
                               (1.0 - gam_n) * (Cm_nm1 - Cmm1_n));
  }

  for (int i = 2, n = 1; i < l - 1; i += 2, ++n) {
    auto gam_n = gam.select(-1, n);
    auto gam_nm1 = gam.select(-1, n - 1);

    auto E1_n = E1.select(-1, n - 1);
    auto E3_n = E3.select(-1, n - 1);

    auto Cp_nm1 = Cp.select(-1, n - 1);
    auto Cpm1_n = Cpm1.select(-1, n);
    auto Cm_nm1 = Cm.select(-1, n - 1);
    auto Cmm1_n = Cmm1.select(-1, n);

    Af_vec.select(-1, i).copy_(2.0 * (1.0 - gam_nm1 * gam_nm1));
    Bf_vec.select(-1, i).copy_((E1_n - E3_n) * (1.0 + gam_n));
    Cf_vec.select(-1, i).copy_((E1_n + E3_n) * (gam_n - 1.0));
    Df_vec.select(-1, i).copy_(E3_n * (Cpm1_n - Cp_nm1) +
                               E1_n * (Cm_nm1 - Cmm1_n));
  }

  Af_vec.select(-1, l - 1).copy_(E1.select(-1, nlay - 1) -
                                 a_surf_in * E3.select(-1, nlay - 1));
  Bf_vec.select(-1, l - 1).copy_(E2.select(-1, nlay - 1) -
                                 a_surf_in * E4.select(-1, nlay - 1));
  Cf_vec.select(-1, l - 1).fill_(0.0);
  Df_vec.select(-1, l - 1).copy_(Bsurf - Cp.select(-1, nlay - 1) +
                                 a_surf_in * Cm.select(-1, nlay - 1));

  // Fill output fluxes
  flx_up = torch::zeros_like(be);
  flx_down = torch::zeros_like(be);

  return out;
}
