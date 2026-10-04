// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <filesystem>

// harp
#include <harp/radiation/bbflux.hpp>
#include <harp/radiation/radiation.hpp>
#include <harp/radiation/radiation_band.hpp>
#include <harp/rtsolver/toon_mckay89.hpp>

// tests
#include "device_testing.hpp"

using namespace harp;

TEST(ToonConfig, from_yaml_reads_toon_options) {
  auto yaml_path =
      std::filesystem::path(__FILE__).parent_path() / "toon_test.yaml";
  auto rad = harp::RadiationOptionsImpl::from_yaml(yaml_path.string());
  ASSERT_EQ(rad->bands().size(), 2u);

  auto const& op = rad->bands().front();

  ASSERT_EQ(op->solver_name(), "toon");
  ASSERT_NE(op->toon(), nullptr);
  EXPECT_EQ(op->toon()->flags(),
            "planck,zenith_correction,hard_surface,delta_eddington_lw");
  EXPECT_TRUE(op->toon()->planck());
  EXPECT_TRUE(op->toon()->zenith_correction());
  EXPECT_EQ(op->toon()->top_emission_flag(), -1);
  EXPECT_TRUE(op->toon()->hard_surface());
  EXPECT_TRUE(op->toon()->delta_eddington_lw());
  EXPECT_EQ(op->toon()->wave_lower(),
            (std::vector<double>{200.0, 200.0, 200.0}));
  EXPECT_EQ(op->toon()->wave_upper(),
            (std::vector<double>{2000.0, 2000.0, 2000.0}));
}

TEST(ToonConfig, radiation_band_registers_solver_module) {
  auto op = harp::RadiationBandOptionsImpl::create();
  op->name("B_toon");
  op->solver_name("toon");
  op->toon(harp::ToonMcKay89OptionsImpl::create());
  op->toon()->flags("planck");
  op->nwave(2);
  op->ncol(1);
  op->nlyr(3);
  op->wavenumber({300.0, 900.0});
  op->weight({600.0, 600.0});
  op->set_wave_lower({0.0, 600.0});
  op->set_wave_upper({600.0, 1200.0});

  harp::RadiationBand band(op);

  EXPECT_NO_THROW({ (void)band->named_modules()["solver"]; });
}

TEST(ToonConfig, planck_flag_controls_thermal_emission) {
  auto wave_lower = std::vector<double>{200.0, 500.0};
  auto wave_upper = std::vector<double>{500.0, 1000.0};

  auto sw_op = harp::ToonMcKay89OptionsImpl::create();
  sw_op->wave_lower(wave_lower);
  sw_op->wave_upper(wave_upper);
  harp::ToonMcKay89 sw_toon(sw_op);

  auto lw_op = harp::ToonMcKay89OptionsImpl::create();
  lw_op->wave_lower(wave_lower);
  lw_op->wave_upper(wave_upper);
  lw_op->flags("planck");
  harp::ToonMcKay89 lw_toon(lw_op);

  auto prop = torch::zeros({2, 1, 3, 3}, torch::kFloat64);
  prop.select(-1, 0).fill_(0.2);
  auto temf = torch::ones({1, 4}, torch::kFloat64) * 300.0;
  std::map<std::string, torch::Tensor> sw_bc;
  std::map<std::string, torch::Tensor> lw_bc;

  auto sw_result = sw_toon(prop, &sw_bc, /*band=*/"", temf);
  auto lw_result = lw_toon(prop, &lw_bc, /*band=*/"", temf);

  EXPECT_TRUE(torch::allclose(sw_result, torch::zeros_like(sw_result)));
  EXPECT_GT(torch::max(torch::abs(lw_result)).item<double>(), 0.0);
}

TEST_P(DeviceTest, simple_toon_mckay89) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({200., 500., 1000.});
  op->wave_upper({500., 1000., 2000.});

  op->report(std::cout);
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  int nwave = op->wave_lower().size();
  int nlyr = 10;
  int ncol = 2;
  int nprop = 3;

  double tau = 0.1;
  double fbeam = 1.0;
  double umu0 = 0.5;
  double tem_K = 300.0;

  auto prop = torch::zeros({nwave, ncol, nlyr, nprop},
                           torch::device(device).dtype(dtype));
  prop.select(-1, 0) = tau;
  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = torch::ones({nwave, ncol}, prop.options()) * fbeam;
  bc["umu0"] = torch::ones({ncol}, prop.options()) * umu0;
  bc["albedo"] = torch::zeros({nwave, ncol}, prop.options());

  for (auto [w0, g] : {std::make_pair(0.1, 0.5), std::make_pair(0.5, 0.5),
                       std::make_pair(0.9, 0.5)}) {
    std::cout << "w0 = " << w0 << ", g = " << g << "\n";

    prop.select(-1, 1) = w0;  // single scattering albedo
    prop.select(-1, 2) = g;   // asymmetry parameter

    auto sw_flx = toon(prop, &bc);

    std::cout << "sw_flx_up = " << sw_flx.select(-1, 0) << "\n";
    std::cout << "sw_flx_dn = " << sw_flx.select(-1, 1) << "\n";
  }

  auto temf = torch::ones({ncol, nlyr + 1}, prop.options()) * tem_K;
  op->flags("planck");

  for (auto [w0, g] : {std::make_pair(0.1, 0.5), std::make_pair(0.5, 0.5),
                       std::make_pair(0.9, 0.5)}) {
    std::cout << "w0 = " << w0 << ", g = " << g << "\n";

    prop.select(-1, 1) = w0;  // single scattering albedo
    prop.select(-1, 2) = g;   // asymmetry parameter
    auto lw_flx = toon(prop, &bc, /*band=*/"", temf);

    std::cout << "lw_flx_up = " << lw_flx.select(-1, 0) << "\n";
    std::cout << "lw_flx_dn = " << lw_flx.select(-1, 1) << "\n";
  }
}

// Regression: for a Lambertian surface the reflected upward flux must equal
// albedo * (total downward flux at the surface), where the total includes both
// the diffuse and the direct beam.  Output level index 0 is the surface.  This
// guards the shortwave bottom boundary condition, including the zero-single-
// scattering-albedo case, which must NOT take the direct-beam-only shortcut
// when the surface albedo is nonzero.
TEST_P(DeviceTest, shortwave_surface_albedo_reflection) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({200., 500., 1000.});
  op->wave_upper({500., 1000., 2000.});

  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  int nwave = op->wave_lower().size();
  int nlyr = 20;
  int ncol = 2;
  int nprop = 3;

  double total_tau = 1.0;
  double fbeam = 100.0;
  double umu0 = 0.5;
  double albedo = 0.1;

  auto prop = torch::zeros({nwave, ncol, nlyr, nprop},
                           torch::device(device).dtype(dtype));
  prop.select(-1, 0) = total_tau / nlyr;  // uniform dtau per layer

  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = torch::ones({nwave, ncol}, prop.options()) * fbeam;
  bc["umu0"] = torch::ones({ncol}, prop.options()) * umu0;
  bc["albedo"] = torch::ones({nwave, ncol}, prop.options()) * albedo;

  double tol = (dtype == torch::kFloat64) ? 1e-11 : 1e-3;

  // ssa = 0 (pure absorption, exercises the shortcut guard), a tiny nonzero
  // ssa, and a genuinely scattering case handled by the general solver.
  for (auto [w0, g] : {std::make_pair(0.0, 0.0), std::make_pair(1.0e-10, 0.0),
                       std::make_pair(0.5, 0.5)}) {
    prop.select(-1, 1) = w0;  // single scattering albedo
    prop.select(-1, 2) = g;   // asymmetry parameter

    auto flx = toon(prop, &bc);
    auto up_surf = flx.select(-1, 0).select(-1, 0);  // up flux at the surface
    auto dn_surf = flx.select(-1, 1).select(-1, 0);  // down flux at the surface

    auto resid = (up_surf - albedo * dn_surf).abs().max().item<double>();
    EXPECT_LT(resid, tol) << "surface reflection failed for w0=" << w0
                          << " g=" << g << " resid=" << resid;
  }
}

// A hard surface of longwave albedo 1 is a mirror: over an isothermal,
// non-scattering column it reflects all the downwelling, so the net flux at the
// surface (output level 0) is zero. A surface that only emits (1 - albedo) B
// and reflects nothing loses the whole downwelling there instead.
TEST_P(DeviceTest, longwave_hard_surface_mirror) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({1.0});
  op->wave_upper({1.0e5});
  op->flags("planck,hard_surface");
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  auto prop = torch::zeros({1, 1, 2, 3}, torch::device(device).dtype(dtype));
  prop.select(-1, 0).fill_(1.0);  // optical depth; w0 = g = 0
  auto temf = torch::full({1, 3}, 1000.0, prop.options());
  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::ones({1, 1}, prop.options());

  auto flx = toon(prop, &bc, /*band=*/"", temf);
  double sigma_t4 = 5.670374419e-8 * 1.0e12;
  double net = (flx[0][0][0][0] - flx[0][0][0][1]).item<double>();
  double tol = (dtype == torch::kFloat64 ? 1e-9 : 1e-5) * sigma_t4;
  EXPECT_LT(std::abs(net), tol) << "net flux at the surface = " << net;
}

// area_scale multiplies the Planck source, and the longwave solve is linear in
// it: a uniform scale of 4 returns 4 times the flux. A scale that is not one
// value per level is refused.
TEST_P(DeviceTest, longwave_area_scale) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({200., 500., 1000.});
  op->wave_upper({500., 1000., 2000.});
  op->flags("planck,hard_surface");
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  int nwave = 3, ncol = 2, nlyr = 10;
  auto prop =
      torch::zeros({nwave, ncol, nlyr, 3}, torch::device(device).dtype(dtype));
  prop.select(-1, 0).fill_(0.1);
  prop.select(-1, 1).fill_(0.5);
  prop.select(-1, 2).fill_(0.5);
  auto temf = torch::linspace(300., 200., nlyr + 1, prop.options())
                  .expand({ncol, nlyr + 1});
  // forward writes into bc, so each call takes a fresh one
  auto solve = [&](torch::Tensor scale) {
    std::map<std::string, torch::Tensor> bc;
    bc["albedo"] = torch::zeros({nwave, ncol}, prop.options());
    if (scale.defined()) bc["area_scale"] = scale;
    return toon(prop, &bc, /*band=*/"", temf);
  };
  auto base = solve(torch::Tensor());
  auto scaled = solve(torch::full({ncol, nlyr + 1}, 4.0, prop.options()));
  double rtol = dtype == torch::kFloat64 ? 1e-12 : 1e-5;
  EXPECT_TRUE(torch::allclose(scaled, 4.0 * base, rtol, 0.0));
  // rank 1, length nlyr+1, is the other documented shape
  auto rank1 = solve(torch::full({nlyr + 1}, 4.0, prop.options()));
  EXPECT_TRUE(torch::allclose(rank1, 4.0 * base, rtol, 0.0));
  // wrong last size, rank 3, and a rank-2 leading size that is not ncol
  EXPECT_THROW(solve(torch::full({ncol, nlyr}, 4.0, prop.options())),
               c10::Error);
  EXPECT_THROW(solve(torch::full({1, ncol, nlyr + 1}, 4.0, prop.options())),
               c10::Error);
  EXPECT_THROW(solve(torch::full({ncol + 1, nlyr + 1}, 4.0, prop.options())),
               c10::Error);
}

// Partial albedo under angle-dependent downwelling. One pure-absorption layer,
// no incoming radiation at the top, so I_down(mu) at the surface varies with
// mu. Surface flux balance holds for a specular mirror and for a Lambertian
// surface. The TOA upward flux does not: this checks it against an independent
// quadrature of the isotropic boundary, which is what fails on the specular
// form in de2a152.
TEST_P(DeviceTest, longwave_lambertian_partial_albedo) {
  const double dtau = 1.0;
  const double albedo = 0.3;
  const double t_surf = 400.0;
  const double t_top = 200.0;
  const double uarr[5] = {0.0985350858, 0.3045357266, 0.5620251898,
                          0.8019865821, 0.9601901429};
  const double wuarr[5] = {0.0157479145, 0.0739088701, 0.1463869871,
                           0.1671746381, 0.0967815902};
  double wsum = 0.0;
  for (double w : wuarr) wsum += w;
  const double twopi = 2.0 * M_PI;

  auto b_of = [](double t) {
    return harp::bbflux_wavenumber(1.0, 1.0e5, torch::tensor(t)).item<double>();
  };
  // BE_IN(0) is the top of the atmosphere, stored last in temf.
  const double b_top = b_of(t_top);
  const double b_surf = b_of(t_surf);
  const double b1 = (b_surf - b_top) / dtau;
  const double alpha1 = twopi * b_top;
  const double alpha2 = twopi * b1;

  double fdn = 0.0;
  double idn[5];
  for (int m = 0; m < 5; ++m) {
    double u = uarr[m];
    double em2 = std::exp(-dtau / u);
    idn[m] = alpha1 * (1.0 - em2) + alpha2 * (u * em2 + dtau - u);
    fdn += idn[m] * wuarr[m];
  }
  const double i_up = twopi * (1.0 - albedo) * b_surf + albedo * fdn / wsum;
  double fup_surf = 0.0;
  double fup_toa = 0.0;
  for (int m = 0; m < 5; ++m) {
    double u = uarr[m];
    double em2 = std::exp(-dtau / u);
    double i_toa =
        i_up * em2 + alpha1 * (1.0 - em2) + alpha2 * (u - (dtau + u) * em2);
    fup_surf += i_up * wuarr[m];
    fup_toa += i_toa * wuarr[m];
  }

  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({1.0});
  op->wave_upper({1.0e5});
  op->flags("planck,hard_surface");
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  auto prop = torch::zeros({1, 1, 1, 3}, torch::device(device).dtype(dtype));
  prop.select(-1, 0).fill_(dtau);
  auto temf = torch::tensor({t_surf, t_top}, prop.options()).view({1, 2});
  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::full({1, 1}, albedo, prop.options());
  auto flx = toon(prop, &bc, /*band=*/"", temf);

  double up_s = flx[0][0][0][0].item<double>();
  double dn_s = flx[0][0][0][1].item<double>();
  double up_toa = flx[0][0][1][0].item<double>();
  double scale = std::abs(fup_toa) + std::abs(fup_surf) + 1.0;
  double tol = (dtype == torch::kFloat64 ? 1e-8 : 1e-4) * scale;
  double balance =
      up_s - ((1.0 - albedo) * twopi * b_surf * wsum + albedo * dn_s);
  EXPECT_LT(std::abs(balance), tol) << "surface balance " << balance;
  EXPECT_LT(std::abs(up_toa - fup_toa), tol)
      << "TOA up flux " << up_toa << " expected " << fup_toa;
}

TEST_P(DeviceTest, longwave_resonance_is_continuous) {
  double constexpr u = 0.5620251898;
  double constexpr w0 = 1.0 - 1.0 / (4.0 * u * u);
  double const relative_offset = dtype == torch::kFloat64 ? 1.0e-7 : 1.0e-3;

  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({1.0});
  op->wave_upper({1.0e5});
  op->flags("planck,hard_surface");
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  auto solve = [&](double scattering_albedo) {
    auto prop =
        torch::zeros({1, 1, 1, 3}, torch::device(device).dtype(dtype));
    prop.select(-1, 0).fill_(1.0);
    prop.select(-1, 1).fill_(scattering_albedo);
    auto temf = torch::tensor({600.0, 300.0}, prop.options()).view({1, 2});
    std::map<std::string, torch::Tensor> bc;
    bc["albedo"] = torch::full({1, 1}, 0.2, prop.options());
    return toon(prop, &bc, /*band=*/"", temf);
  };

  auto exact = solve(w0);
  auto below = solve(w0 * (1.0 - relative_offset));
  auto above = solve(w0 * (1.0 + relative_offset));
  auto reference = 0.5 * (below + above);

  EXPECT_TRUE(torch::all(torch::isfinite(exact.cpu())).item<bool>());
  EXPECT_TRUE(torch::all(torch::isfinite(reference.cpu())).item<bool>());
  double rtol = dtype == torch::kFloat64 ? 2.0e-6 : 5.0e-3;
  double scale = torch::abs(reference).max().item<double>();
  EXPECT_LT(torch::abs(exact - reference).max().item<double>(),
            rtol * (scale + 1.0));
}

TEST_P(DeviceTest, longwave_uncapped_ratio_avoids_overflow) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({1.0});
  op->wave_upper({1.0e5});
  op->flags("planck,hard_surface");
  harp::ToonMcKay89 toon(op);
  toon->to(device, dtype);

  auto prop =
      torch::zeros({1, 1, 1, 3}, torch::device(device).dtype(dtype));
  prop.select(-1, 0).fill_(1000.0);
  prop.select(-1, 1).fill_(0.999999);
  auto temf = torch::tensor({600.0, 300.0}, prop.options()).view({1, 2});
  std::map<std::string, torch::Tensor> bc;
  bc["albedo"] = torch::full({1, 1}, 0.2, prop.options());

  auto result = toon(prop, &bc, /*band=*/"", temf);

  EXPECT_TRUE(torch::all(torch::isfinite(result.cpu())).item<bool>());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
