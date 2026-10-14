// C/C++
#include <cmath>
#include <cstring>
#include <filesystem>

// base
#include <configure.h>

// external
#include <gtest/gtest.h>

// torch
#include <c10/core/InferenceMode.h>
#include <torch/torch.h>

// harp
#include <harp/opacity/molecule_cia.hpp>
#include <harp/opacity/opacity_options.hpp>

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

#include "device_testing.hpp"

namespace fs = std::filesystem;

namespace harp {
extern std::vector<std::string> species_names;
extern std::vector<double> species_weights;
}  // namespace harp

namespace {

#ifdef NETCDFOUTPUT
void check_nc(int status) {
  ASSERT_EQ(status, NC_NOERR) << nc_strerror(status);
}

void put_units(int fileid, int varid, char const* units) {
  check_nc(nc_put_att_text(fileid, varid, "units", std::strlen(units), units));
}

// H2O-H2O table on 5 wavenumbers, 3 pressures and 3 temperature anomalies,
// with a base temperature that changes with pressure, so that every input of
// the interpolation carries a nonzero derivative.
fs::path write_table() {
  auto path = fs::temp_directory_path() / "pyharp_test_cia_autograd.nc";
  double const wave[] = {100.0, 200.0, 300.0, 400.0, 500.0};
  double const pres[] = {1.0e4, 1.0e5, 1.0e6};
  double const temp[] = {280.0, 300.0, 320.0};
  double const del_temp[] = {-50.0, 0.0, 50.0};
  std::vector<double> sigma(5 * 3 * 3);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        sigma[(i * 3 + j) * 3 + k] =
            std::exp(-20.0 + 0.3 * i + 0.2 * j + 0.1 * k + 0.05 * i * k -
                     0.04 * j * k + 0.03 * i * j);
      }
    }
  }

  int fileid = -1, dims[3], var_wave, var_pres, var_del_temp, var_temp, var_cia;
  check_nc(nc_create(path.c_str(), NC_CLOBBER, &fileid));
  check_nc(nc_def_dim(fileid, "wavenumber", 5, &dims[0]));
  check_nc(nc_def_dim(fileid, "pressure", 3, &dims[1]));
  check_nc(nc_def_dim(fileid, "del_temperature", 3, &dims[2]));
  check_nc(nc_def_var(fileid, "wavenumber", NC_DOUBLE, 1, &dims[0], &var_wave));
  check_nc(nc_def_var(fileid, "pressure", NC_DOUBLE, 1, &dims[1], &var_pres));
  check_nc(nc_def_var(fileid, "del_temperature", NC_DOUBLE, 1, &dims[2],
                      &var_del_temp));
  check_nc(
      nc_def_var(fileid, "temperature", NC_DOUBLE, 1, &dims[1], &var_temp));
  check_nc(nc_def_var(fileid, "binary_absorption_coefficient_h2o_h2o",
                      NC_DOUBLE, 3, dims, &var_cia));
  put_units(fileid, var_wave, "cm^-1");
  put_units(fileid, var_pres, "Pa");
  put_units(fileid, var_del_temp, "K");
  put_units(fileid, var_temp, "K");
  put_units(fileid, var_cia, "m^5 mol^-2");
  check_nc(nc_enddef(fileid));
  check_nc(nc_put_var_double(fileid, var_wave, wave));
  check_nc(nc_put_var_double(fileid, var_pres, pres));
  check_nc(nc_put_var_double(fileid, var_del_temp, del_temp));
  check_nc(nc_put_var_double(fileid, var_temp, temp));
  check_nc(nc_put_var_double(fileid, var_cia, sigma.data()));
  check_nc(nc_close(fileid));
  return path;
}

int constexpr kWave = 3, kCol = 2, kLyr = 3;
#endif

}  // namespace

#ifdef NETCDFOUTPUT
// Builds the table, the module and the inputs on one device and dtype.
struct CIASetup {
  torch::Device device;
  torch::Dtype dtype;

  harp::MoleculeCIA make_cia() {
    harp::species_names = {"H2O"};
    harp::species_weights = {18.0e-3};
    auto options = harp::OpacityOptionsImpl::create();
    options->type("molecule-cia")
        .species_ids({0})
        .opacity_files({write_table().string()})
        .warn_out_of_bounds(false);
    harp::MoleculeCIA cia(options);
    cia->to(device, dtype);
    return cia;
  }

  torch::Tensor make(std::vector<double> const& values,
                     std::vector<int64_t> const& shape) {
    return torch::tensor(values, torch::kFloat64)
        .view(shape)
        .to(device, dtype)
        .contiguous();
  }

  torch::Tensor conc() {
    return make({0.2, 0.4, 0.6, 0.8, 1.0, 1.2}, {kCol, kLyr, 1});
  }
  torch::Tensor pres() {
    return make({2.0e4, 5.0e4, 9.0e4, 1.5e5, 3.0e5, 6.0e5}, {kCol, kLyr});
  }
  torch::Tensor temp() {
    return make({305.0, 275.0, 320.0, 330.0, 290.0, 345.0}, {kCol, kLyr});
  }
  torch::Tensor wavenumber() { return make({150.0, 275.0, 420.0}, {kWave}); }

  std::map<std::string, torch::Tensor> atm() {
    return {{"pres", pres()}, {"temp", temp()}, {"wavenumber", wavenumber()}};
  }

  // A fixed weighting, so that no symmetry of a plain sum hides an error.
  torch::Tensor weights(torch::Tensor const& out) {
    return torch::linspace(0.5, 1.5, out.numel(), out.options()).view_as(out);
  }
};

// Concentration-only gradients and ignored inputs are observable API behavior.
TEST_P(DeviceTest, CIAConcentrationGradientAndIgnoredInputs) {
  CIASetup s{device, dtype};
  auto cia = s.make_cia();
  auto expected = cia->forward(s.conc(), s.atm());
  auto c = s.conc().requires_grad_(true);
  auto out = cia->forward(c, s.atm());
  EXPECT_TRUE(torch::equal(out, expected));
  out.sum().backward();
  EXPECT_TRUE(torch::isfinite(c.grad()).all().item<bool>());
  auto a = s.atm();
  a["wavelength"] = (1.0e4 / s.wavenumber()).requires_grad_(true);
  a["unused"] = s.temp().requires_grad_(true);
  auto ignored = cia->forward(s.conc(), a);
  EXPECT_TRUE(torch::equal(ignored, expected));
  EXPECT_FALSE(ignored.requires_grad());
}

// no_grad and inference mode preserve values without recording a graph.
TEST_P(DeviceTest, CIAValuesUnderNoGradAndInferenceMode) {
  CIASetup s{device, dtype};
  auto cia = s.make_cia();
  auto expected = cia->forward(s.conc(), s.atm());
  cia->ln_sigma_binary.requires_grad_(true);
  auto a = s.atm();
  a["temp"].requires_grad_(true);
  a["pres"].requires_grad_(true);
  auto check = [&]() {
    auto out = cia->forward(s.conc(), a);
    EXPECT_TRUE(torch::equal(out, expected));
    EXPECT_FALSE(out.requires_grad());
  };
  {
    torch::NoGradGuard guard;
    check();
  }
  {
    c10::InferenceMode guard;
    check();
  }
  {
    c10::InferenceMode guard;
    torch::AutoGradMode enable(true);
    check();
  }
}

// Wavelength is the distinct converted spectral dependency. Other query,
// coordinate, table and nonleaf dependencies are checked numerically below.
TEST_P(DeviceTest, CIAWavelengthGradient) {
  CIASetup s{device, dtype};
  auto cia = s.make_cia();
  auto a = s.atm();
  a.erase("wavenumber");
  auto wavelength = (1.0e4 / s.wavenumber()).requires_grad_(true);
  a["wavelength"] = wavelength;
  auto out = cia->forward(s.conc(), a);
  auto actual = torch::autograd::grad({out.sum()}, {wavelength})[0];
  auto wave = s.wavenumber().requires_grad_(true);
  a.erase("wavelength");
  a["wavenumber"] = wave;
  auto dw = torch::autograd::grad({cia->forward(s.conc(), a).sum()}, {wave})[0];
  EXPECT_TRUE(torch::allclose(actual, dw * (-1.0e4 / wavelength.square())));
}

// A wavenumber takes precedence over a wavelength; a wavelength alone is
// converted to wavenumber; without either the table's own grid is used.
TEST_P(DeviceTest, CIASpectralQueryPrecedence) {
  CIASetup s{device, dtype};
  auto cia = s.make_cia();
  auto a = s.atm();
  auto by_wavenumber = cia->forward(s.conc(), a);

  auto both = s.atm();
  both["wavelength"] = s.make({7.0, 8.0, 9.0}, {kWave});
  EXPECT_TRUE(torch::equal(cia->forward(s.conc(), both), by_wavenumber));

  auto wavelength = s.make({50.0, 40.0, 25.0}, {kWave});
  auto by_wavelength = s.atm();
  by_wavelength.erase("wavenumber");
  by_wavelength["wavelength"] = wavelength;
  auto converted = s.atm();
  converted["wavenumber"] = 1.0e4 / wavelength;
  EXPECT_TRUE(torch::equal(cia->forward(s.conc(), by_wavelength),
                           cia->forward(s.conc(), converted)));

  auto table_grid = s.atm();
  table_grid.erase("wavenumber");
  EXPECT_EQ(cia->forward(s.conc(), table_grid).size(0), 5);
}

// Expanding the queries changes the graph, not the forward values.
TEST_P(DeviceTest, CIAForwardDoesNotDependOnTheRoute) {
  CIASetup s{device, dtype};
  auto cia = s.make_cia();
  auto compact = cia->forward(s.conc(), s.atm()).detach();
  auto a = s.atm();
  a["temp"].requires_grad_(true);
  auto expanded = cia->forward(s.conc(), a).detach();
  EXPECT_TRUE(torch::equal(compact, expanded));
}

// Reverse-mode derivatives with respect to every differentiable input: the
// atmosphere, the spectral query, the table coordinates, the base
// temperature and the table itself. Checked against central differences in
// float64, and against the float64 gradient in float32.
TEST_P(DeviceTest, CIAGradientsOfAllInterpolationInputs) {
  CIASetup s{device, dtype};
  std::vector<std::string> const names = {"conc",
                                          "pres",
                                          "temp",
                                          "wavenumber query",
                                          "wavenumber",
                                          "ln_pressure",
                                          "temperature_anomaly",
                                          "ln_temperature_base",
                                          "ln_sigma_binary"};

  // Loss and the tensors it is differentiated against, for one dtype.
  auto evaluate = [&](torch::Dtype dt,
                      std::map<std::string, torch::Tensor> shift) {
    auto saved = s.dtype;
    s.dtype = dt;
    auto cia = s.make_cia();
    std::map<std::string, torch::Tensor> leaves = {
        {"conc", s.conc()},
        {"pres", s.pres()},
        {"temp", s.temp()},
        {"wavenumber query", s.wavenumber()}};
    auto buffers = cia->named_buffers();
    for (auto const& name : names) {
      if (leaves.count(name) == 0) leaves[name] = buffers[name];
    }
    {
      torch::NoGradGuard no_grad;
      for (auto& [name, delta] : shift) leaves[name].add_(delta.to(dt));
    }
    for (auto& [name, t] : leaves) t.requires_grad_(true);
    std::map<std::string, torch::Tensor> a = {
        {"pres", leaves["pres"]},
        {"temp", leaves["temp"]},
        {"wavenumber", leaves["wavenumber query"]}};
    auto out = cia->forward(leaves["conc"], a);
    auto loss = (out * s.weights(out)).sum();
    std::vector<torch::Tensor> inputs;
    for (auto const& name : names) inputs.push_back(leaves[name]);
    auto grads = torch::autograd::grad({loss}, inputs);
    s.dtype = saved;
    return std::make_pair(loss.item<double>(), grads);
  };

  auto [loss, grads] = evaluate(dtype, {});
  for (size_t i = 0; i < names.size(); ++i) {
    ASSERT_TRUE(grads[i].defined()) << names[i];
    EXPECT_TRUE(torch::all(torch::isfinite(grads[i])).item<bool>()) << names[i];
  }

  if (dtype == torch::kFloat32) {
    auto reference = evaluate(torch::kFloat64, {}).second;
    for (size_t i = 0; i < names.size(); ++i) {
      auto g = grads[i].to(torch::kFloat64);
      EXPECT_TRUE(
          torch::allclose(g, reference[i], 1.0e-4,
                          1.0e-4 * reference[i].abs().max().item<double>()))
          << names[i];
    }
    return;
  }

  // One random direction per input; the step is small next to the input so
  // that no query crosses a table node.
  std::map<std::string, double> const size = {{"conc", 1.0},
                                              {"pres", 1.0e5},
                                              {"temp", 300.0},
                                              {"wavenumber query", 300.0},
                                              {"wavenumber", 300.0},
                                              {"ln_pressure", 10.0},
                                              {"temperature_anomaly", 50.0},
                                              {"ln_temperature_base", 6.0},
                                              {"ln_sigma_binary", 20.0}};
  torch::manual_seed(167);
  for (size_t i = 0; i < names.size(); ++i) {
    auto direction = torch::randn_like(grads[i]);
    double const step = 1.0e-7 * size.at(names[i]);
    auto plus = evaluate(dtype, {{names[i], step * direction}}).first;
    auto minus = evaluate(dtype, {{names[i], -step * direction}}).first;
    auto numeric = (plus - minus) / (2.0 * step);
    auto analytic = (grads[i] * direction).sum().item<double>();
    EXPECT_NEAR(analytic, numeric, 1.0e-5 * std::abs(numeric)) << names[i];
  }
}

// A non-leaf temperature passes its gradient on unchanged.
TEST_P(DeviceTest, CIANonleafInputs) {
  CIASetup s{device, dtype};
  auto cia = s.make_cia();

  auto leaf = s.temp().requires_grad_(true);
  auto a = s.atm();
  a["temp"] = leaf;
  auto out = cia->forward(s.conc(), a);
  auto direct =
      torch::autograd::grad({(out * s.weights(out)).sum()}, {leaf})[0];

  auto root = s.temp().requires_grad_(true);
  a["temp"] = root * 1.0;
  out = cia->forward(s.conc(), a);
  auto through =
      torch::autograd::grad({(out * s.weights(out)).sum()}, {root})[0];
  EXPECT_TRUE(torch::equal(direct, through));
}

// Second derivatives through create_graph, d/dT and d/dc of u . dL/dT:
// against central differences of dL/dT in float64, and against the float64
// result in float32.
TEST_P(DeviceTest, CIASecondDerivatives) {
  auto second_derivatives = [](CIASetup s) {
    auto cia = s.make_cia();
    auto t = s.temp().requires_grad_(true);
    auto c = s.conc().requires_grad_(true);
    auto u = torch::linspace(-1.0, 2.0, t.numel(), t.options()).view_as(t);
    auto b = s.atm();
    b["temp"] = t;
    auto o = cia->forward(c, b);
    auto g = torch::autograd::grad({(o * s.weights(o)).sum()}, {t}, {}, true,
                                   true)[0];
    return torch::autograd::grad({(g * u).sum()}, {t, c});
  };

  CIASetup s{device, dtype};
  auto second = second_derivatives(s);
  for (auto const& d : second) {
    ASSERT_TRUE(d.defined());
    EXPECT_TRUE(torch::all(torch::isfinite(d)).item<bool>());
  }

  if (dtype == torch::kFloat32) {
    auto reference = second_derivatives(CIASetup{device, torch::kFloat64});
    for (int which = 0; which < 2; ++which) {
      auto r = reference[which];
      EXPECT_TRUE(torch::allclose(second[which].to(torch::kFloat64), r, 1.0e-3,
                                  1.0e-3 * r.abs().max().item<double>()))
          << (which == 0 ? "d2/dT2" : "d2/dT dc");
    }
    return;
  }

  // u . dL/dT at shifted temperature or concentration.
  auto cia = s.make_cia();
  auto t = s.temp();
  auto c = s.conc();
  auto u = torch::linspace(-1.0, 2.0, t.numel(), t.options()).view_as(t);
  auto projected_first = [&](torch::Tensor tt, torch::Tensor const& cc) {
    tt.requires_grad_(true);
    auto b = s.atm();
    b["temp"] = tt;
    auto o = cia->forward(cc, b);
    auto g = torch::autograd::grad({(o * s.weights(o)).sum()}, {tt})[0];
    return (g * u).sum().item<double>();
  };

  torch::manual_seed(168);
  for (int which = 0; which < 2; ++which) {
    auto direction = torch::randn_like(which == 0 ? t : c);
    double const step = which == 0 ? 1.0e-4 : 1.0e-6;
    auto shifted = [&](double sign) {
      auto delta = sign * step * direction;
      return which == 0 ? projected_first(t + delta, c)
                        : projected_first(t.clone(), c + delta);
    };
    auto numeric = (shifted(1.0) - shifted(-1.0)) / (2.0 * step);
    auto analytic = (second[which] * direction).sum().item<double>();
    EXPECT_NEAR(analytic, numeric, 1.0e-5 * std::abs(numeric) + 1.0e-20)
        << (which == 0 ? "d2/dT2" : "d2/dT dc");
  }
}

#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
