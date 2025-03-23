// C/C++
#include <stdexcept>

// harp
#include <math/tridiag.hpp>

#include "toon_mckay89.hpp"

torch::Tensor ToonMcKay89Impl::shortwave_solver(
    torch::Tensor F0_in, torch::Tensor mu_in, torch::Tensor tau_in,
    torch::Tensor w_in, torch::Tensor g_in, torch::Tensor w_surf_in) {
  int nwave = tau_in.size(0);
  int ncol = tau_in.size(1);
  int nlay = tau_in.size(2);

  // Input validation
  if (mu_in.size(0) != ncol || w_in.size(-1) != nlay ||
      g_in.size(-1) != nlay {
    throw std::invalid_argument("Input vectors have incorrect sizes.");
  }

  // increase the last dimension by 1 (lyr -> lvl)
  auto shape = tau_in.sizes().vec();
  shape.back() += 1;
  torch::Tensor tau_cum = torch::zeros(shape, tau_in.options());
  tau_cum.narrow(-1, 1, nlay) = tau_in.cumsum(-1);

  int nlev = tau_cum.size(-1);

  // Initialize output flux arrays
  auto out = torch::zeros({ncol, nlev, 2}, tau_cum.options());
  flx_down = out.select(-1, 0);
  flx_up = out.select(-1, 1);

  // Constants
  const double sqrt3 = std::sqrt(3.0);
  const double sqrt3d2 = sqrt3 / 2.0;
  const double bsurf = 0.0;
  const double btop = 0.0;

  // Check if all single scattering albedos are effectively zero
  bool all_w0_zero = (w_in <= 1.0e-12).all().item<bool>();

  if (all_w0_zero) {  // no scattering
    // Direct beam only
    // No zenith correction, use regular method
    if (!options.zenith_correction) {
      flx_down = F0_in.unsqueeze(-1) * mu_in.unsqueeze(-1) *
                 (-tau_cum / mu_in.unsqueeze(-1)).exp();
    } else {
      // Zenith angle correction using cumulative transmission
      TORCH_CHECK(mu_in.size(-1) == nlay,
                  "The last dimension of mu_in should have layers");
      auto trans_cum = torch::zeros_like(tau_cum);
      trans_cum.narrow(-1, 1, nlay) = tau_in / mu_in;
      trans_cum.narrow(-1, 1, nlay) = torch::cumsum(trans_cum, -1);

      flx_down = F0_in.unsqueeze(-1) * mu_in * torch::exp(-trans_cum);
    }

    // Adjust the downward flux at the surface layer for surface albedo
    flx_down.select(-1, nlev - 1) *= 1.0 - w_surf_in;

    // Upward flux remains zero
    return out;
  }

  // Delta Eddington scaling
  auto w0 = ((1.0 - g_in * g_in) * w_in) / (1.0 - w_in * g_in * g_in);
  auto dtau = (1.0 - w_in * g_in * g_in) * tau_in;
  auto hg = g_in / (1.0 + g_in);

  // Initialize tau_total
  torch::Tensor tau_total = torch::zeros_like(tau_cum);
  tau_total.narrow(-1, 1, nlay) = dtau.cumsum(-1);

  // Compute g1, g2, g3, g4
  auto g1 = sqrt3d2 * (2.0 - w0 * (1.0 + hg));
  auto g2 = sqrt3d2 * w0 * (1.0 - hg);
  // Prevent division by zero
  g2.clamp_(1.0e-10);

  // Compute mu_zm at midpoints
  auto mu_zm = (mu_in.narrow(-1, 0, nlay) + mu_in.narrow(-1, 1, nlay)) / 2.0;
  auto g3 = (1.0 - sqrt3 * hg * mu_zm) / 2.0;
  auto g4 = 1.0 - g3;

  // Compute lam and gam
  auto lam = (g1 * g1 - g2 * g2).square();
  auto gam = (g1 - lam) / g2;

  // Compute denom and handle denom == 0
  auto denom = lam * lam - (1.0 / mu_in.select(-1, nlev - 1).square());
  denom.clamp_(1.0e-10);

  // Compute Am and Ap
  auto Am = F0_in * w0 * (g4 * (g1 + 1.0 / mu_in.select(-1, nlev - 1)) + g2 * g3) / denom;
  auto Ap = F0_in * w0 * (g3 * (g1 - 1.0 / mu_in.select(-1, nlev - 1)) + g2 * g4) / denom;

  // Compute Cpm1 and Cmm1 at the top of the layer
  auto Cpm1 = Ap * (-tau_total.narrow(-1, 0, nlay) / mu_in.select(-1, nlev - 1)).exp();
  auto Cmm1 = Am * (-tau_total.narrow(-1, 0, nlay) / mu_in.select(-1, nlev - 1)).exp();

  // Compute Cp and Cm at the bottom of the layer
  auto Cp = Ap * (-tau_total.narrow(-1, 1, nlay) / mu_in.select(-1, nlev - 1)).exp();
  auto Cm = Am * (-tau_total.narrow(-1, 1, nlay) / mu_in.select(-1, nlev - 1)).exp();

  // Compute exponential terms, clamped to prevent overflow
  auto exptrm = (lam * dtau).clamp_(35.0);
  auto Ep = exptrm.exp();
  auto Em = 1.0 / Ep;
  auto E1 = Ep + gam * Em;
  auto E2 = Ep - gam * Em;
  auto E3 = gam * Ep + Em;
  auto E4 = gam * Ep - Em;

  // Initialize Af, Bf, Cf, Df
  int l = 2 * nlay;
  auto Af = torch::zeros({nwave, ncol, l}, tau_in.options());
  auto Bf = torch::zeros({nwave, ncol, l}, tau_in.options());
  auto Cf = torch::zeros({nwave, ncol, l}, tau_in.options());
  auto Df = torch::zeros({nwave, ncol, l}, tau_in.options());

  // Boundary conditions at the top
  Af.select(-1, 0) = 0.0;
  Bf.select(-1, 0) = gam.select(-1, 0) + 1.0;
  Cf.select(-1, 0) = gam.select(-1, 0) - 1.0;
  Df.select(-1, 0) = btop - Cmm1.select(-1, 0);
  for (int i = 1, n = 1; i < l - 1; i += 2, ++n) {
    TORCK_CHECK(n < nlay,
                "Index out of range in sw_Toon89 Af, Bf, Cf, Df population.");

    Af.select(-1, i) = (E1.select(-1, n - 1) + E3.select(-1, n - 1)) *
                       (gam.select(-1, n) - 1.0);
    Bf.select(-1, i) = (E2.select(-1, n - 1) + E4.select(-1, n - 1)) *
                       (gam.select(-1, n) - 1.0);
    Cf.select(-1, i) = 2.0 * (1.0 - gam.select(-1, n).square());
    Df.select(-1, i) =
        (gam.select(-1, n) - 1.0) *
            (Cpm1.select(-1, n) - Cp.select(-1, n - 1)) +
        (1.0 - gam.select(-1, n)) * (Cm.select(-1, n - 1) - Cmm1.select(-1, n));
  }

  // Populate Af, Bf, Cf, Df for even indices
  // Start from n=1 to avoid negative indexing (Cp(n-1) when n=0)
  for (int i = 2, n = 1; i < l - 1; i += 2, ++n) {
    TORCH_CHECK(n < nlay,
                "Index out of range in sw_Toon89 Af, Bf, Cf, Df population.");

    Af.select(-1, i) = 2.0 * (1.0 - gam.select(-1, n).square());
    Bf.select(-1, i) = (E1.select(-1, n - 1) - E3.select(-1, n - 1)) *
                       (1.0 + gam.select(-1, n));
    Cf.select(-1, i) = (E1.select(-1, n - 1) + E3.select(-1, n - 1)) *
                       (gam.select(-1, n) - 1.0);
    Df.select(-1, i) =
        E3.select(-1, n - 1) * (Cpm1.select(-1, n) - Cp.select(-1, n - 1)) +
        E1.select(-1, n - 1) * (Cm.select(-1, n - 1) - Cmm1.select(-1, n));
  }

  // Boundary conditions at l (last index)
  Af.select(-1, l - 1) = E1.select(-1, nlay - 1) - w_surf_in * E3.select(-1, nlay - 1);
  Bf.select(-1, l - 1) = E2.select(-1, nlay - 1) - w_surf_in * E4.select(-1, nlay - 1);
  Cf.select(-1, l - 1) = 0.0;
  Df.select(-1, l - 1) = bsurf - Cp.select(-1, nlay - 1) + w_surf_in * Cm.select(-1, nlay - 1);

  // Solve the tridiagonal system
  tridiag_lu(Af, Bf, Cf);
  tridiag_solve(Df, Af, Bf, Cf);

  // Compute xk1 and xk2 from xk
  // select even and odd indices
  auto xk_2n = Df.index_select(-1, torch::arange(0, tensor.size(0), 2));
  auto xk_2np1 = Df.index_select(-1, torch::arange(1, tensor.size(0), 2));

  auto xk1 = xk_2n + xk_2np1;
  auto xk2 = xk_2n - xk_2np1;

  xk2 = torch::where(torch::abs(xk2 / xk_2n) < 1e-30, torch::zeros_like(xk2), xk2);

  // Populate flx_up and flx_down for layers 1 to nlay
  flx_up.select(-1, 0, nlay) = xk1 + g * xk2 + Cpm1;
  flx_down.select(-1, 0, nlay) = xk1 * gam + xk2 + Cmm1;

  // Compute flx_up and flx_down at level nlev
  flx_up.select(-1, 0, nlev - 1) = xk1.select(-1, nlay - 1) * std::exp(1.0)
    + gam.select(-1, nlay - 1) * xk2.select(-1, nlay - 1) * std::exp(-1.0)
    + Cp.select(-1, nlay - 1);
  flx_down.select(-1, 0, nlev - 1) = xk1.select(-1, nlay - 1) * std::exp(1.0)
    * gam.select(-1, nlay - 1) + xk2.select(-1, nlay - 1) * std::exp(-1.0)
    + Cm.select(-1, nlay - 1);

  // Compute dir flux
  Torch::Tensor dir;
  if (!options.zenith_correction) {
    // No zenith correction
    dir = F0_in.unsqueeze(-1) * mu_in.unsqueeze(-1) *
          (-tau_cum / mu_in.unsqueeze(-1)).exp();
  } else {
    // Zenith angle correction
    TORCH_CHECK(mu_in.size(-1) == nlay,
                "The last dimension of mu_in should have layers");
    auto trans_cum = torch::zeros_like(tau_cum);
    trans_cum.narrow(-1, 1, nlay) = tau_in / mu_in;
    trans_cum.narrow(-1, 1, nlay) = torch::cumsum(trans_cum, -1);

    dir = F0_in * mu_in * torch::exp(-trans_cum);
  }

  // Adjust the downward flux at the surface layer for surface albedo
  dir.select(-1, nlev - 1) *= 1.0 - w_surf_in;

  // for(int i=0; i <nlev; ++i) std::cout << "flux_up: " << flx_up(i) << "
  // flux_down: " << flx_down(i) << " dirflux_down: " << dir(i) << std::endl;
  //  Add the direct beam contribution
  flx_down += dir;

  // Ensure no negative fluxes due to numerical errors
  out.clamp_(0.0);

  return out;
}
