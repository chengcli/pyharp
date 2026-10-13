// C/C++
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

// base
#include <configure.h>

// external
#include <gtest/gtest.h>

// harp
#include <harp/constants.h>
#include <harp/math/interpolation.hpp>

#include <harp/opacity/molecule_cia.hpp>
#include <harp/opacity/molecule_line.hpp>
#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/respq_table.hpp>
#include <harp/radiation/radiation_band.hpp>
#include <harp/rtsolver/toon_mckay89.hpp>

// netcdf
#ifdef NETCDFOUTPUT
extern "C" {
#include <netcdf.h>
}
#endif

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

void put_text_attr(int fileid, int varid, char const* name, char const* value) {
  check_nc(nc_put_att_text(fileid, varid, name, std::strlen(value), value));
}

fs::path write_test_dataset(bool split_continuum = false,
                            bool celsius_temperatures = false,
                            char const* del_temperature_units = nullptr) {
  auto path = fs::temp_directory_path() /
              (celsius_temperatures ? "pyharp_test_molecule_celsius.nc"
               : split_continuum    ? "pyharp_test_molecule_line_split.nc"
                                    : "pyharp_test_molecule_line.nc");
  int fileid = -1;
  check_nc(nc_create(path.c_str(), NC_CLOBBER, &fileid));

  int dim_del_temp = -1, dim_pressure = -1, dim_wavenumber = -1;
  check_nc(nc_def_dim(fileid, "del_temperature", 2, &dim_del_temp));
  check_nc(nc_def_dim(fileid, "pressure", 2, &dim_pressure));
  check_nc(nc_def_dim(fileid, "wavenumber", 3, &dim_wavenumber));

  int var_wavenumber = -1, var_pressure = -1, var_del_temp = -1;
  int var_temperature = -1, var_line = -1, var_cont = -1, var_cia = -1;
  int var_cont_self = -1, var_cont_foreign = -1;
  check_nc(nc_def_var(fileid, "wavenumber", NC_DOUBLE, 1, &dim_wavenumber,
                      &var_wavenumber));
  check_nc(nc_def_var(fileid, "pressure", NC_DOUBLE, 1, &dim_pressure,
                      &var_pressure));
  check_nc(nc_def_var(fileid, "del_temperature", NC_DOUBLE, 1, &dim_del_temp,
                      &var_del_temp));
  check_nc(nc_def_var(fileid, "temperature", NC_DOUBLE, 1, &dim_pressure,
                      &var_temperature));

  int dims3[3] = {dim_del_temp, dim_pressure, dim_wavenumber};
  check_nc(
      nc_def_var(fileid, "sigma_line_h2o", NC_DOUBLE, 3, dims3, &var_line));
  check_nc(nc_def_var(fileid, "sigma_continuum_h2o_mt_ckd", NC_DOUBLE, 3, dims3,
                      &var_cont));
  if (split_continuum) {
    check_nc(nc_def_var(fileid, "sigma_continuum_h2o_self_mt_ckd", NC_DOUBLE, 3,
                        dims3, &var_cont_self));
    check_nc(nc_def_var(fileid, "sigma_continuum_h2o_foreign_mt_ckd", NC_DOUBLE,
                        3, dims3, &var_cont_foreign));
  }
  check_nc(nc_def_var(fileid, "binary_absorption_coefficient_h2_he", NC_DOUBLE,
                      3, dims3, &var_cia));

  put_text_attr(fileid, var_wavenumber, "units", "cm^-1");
  put_text_attr(fileid, var_pressure, "units", "Pa");
  put_text_attr(fileid, var_del_temp, "units",
                del_temperature_units  ? del_temperature_units
                : celsius_temperatures ? "degC"
                                       : "K");
  put_text_attr(fileid, var_temperature, "units",
                celsius_temperatures ? "degC" : "K");
  put_text_attr(fileid, var_line, "units", "cm^2 molecule^-1");
  put_text_attr(fileid, var_cont, "units", "cm^2 molecule^-1");
  if (split_continuum) {
    put_text_attr(fileid, var_cont_self, "units", "cm^2 molecule^-1");
    put_text_attr(fileid, var_cont_foreign, "units", "cm^2 molecule^-1");
  }
  put_text_attr(fileid, var_cia, "units", "cm^5 molecule^-2");

  check_nc(nc_enddef(fileid));

  double const wavenumber[] = {20.0, 21.0, 22.0};
  double const pressure[] = {1.0e5, 1.0e6};
  double const del_temp[] = {-10.0, 10.0};
  double const temperature[] = {celsius_temperatures ? 26.85 : 300.0,
                                celsius_temperatures ? 226.85 : 500.0};

  std::vector<double> sigma_line(2 * 2 * 3);
  std::vector<double> sigma_cont(2 * 2 * 3);
  std::vector<double> sigma_cont_self(2 * 2 * 3, 2.0e-24);
  std::vector<double> sigma_cont_foreign(2 * 2 * 3, 6.0e-24);
  std::vector<double> sigma_cia(2 * 2 * 3);

  for (int idt = 0; idt < 2; ++idt) {
    for (int ip = 0; ip < 2; ++ip) {
      for (int iw = 0; iw < 3; ++iw) {
        auto idx = (idt * 2 + ip) * 3 + iw;
        sigma_line[idx] = (1.0 + idx) * 1.0e-24;
        sigma_cont[idx] = (0.1 + idx) * 1.0e-24;
        sigma_cia[idx] = (2.0 + idx) * 1.0e-46;
      }
    }
  }

  sigma_line[0] = 0.0;
  sigma_cont[0] = 0.0;
  sigma_cia[0] = 0.0;

  check_nc(nc_put_var_double(fileid, var_wavenumber, wavenumber));
  check_nc(nc_put_var_double(fileid, var_pressure, pressure));
  check_nc(nc_put_var_double(fileid, var_del_temp, del_temp));
  check_nc(nc_put_var_double(fileid, var_temperature, temperature));
  check_nc(nc_put_var_double(fileid, var_line, sigma_line.data()));
  check_nc(nc_put_var_double(fileid, var_cont, sigma_cont.data()));
  if (split_continuum) {
    check_nc(nc_put_var_double(fileid, var_cont_self, sigma_cont_self.data()));
    check_nc(
        nc_put_var_double(fileid, var_cont_foreign, sigma_cont_foreign.data()));
  }
  check_nc(nc_put_var_double(fileid, var_cia, sigma_cia.data()));
  check_nc(nc_close(fileid));

  return path;
}

fs::path write_respq_dataset() {
  auto path = fs::temp_directory_path() / "pyharp_test_respq.nc";
  int fileid = -1;
  check_nc(nc_create(path.c_str(), NC_CLOBBER, &fileid));

  int point = -1, pressure = -1, offset = -1, species = -1;
  int linear = -1, binary = -1, moment = -1;
  check_nc(nc_def_dim(fileid, "point", 2, &point));
  check_nc(nc_def_dim(fileid, "pressure", 2, &pressure));
  check_nc(nc_def_dim(fileid, "temperature_offset", 2, &offset));
  check_nc(nc_def_dim(fileid, "species", 2, &species));
  check_nc(nc_def_dim(fileid, "linear_component", 1, &linear));
  check_nc(nc_def_dim(fileid, "binary_component", 1, &binary));
  check_nc(nc_def_dim(fileid, "moment", 2, &moment));

  auto define_1d = [&](char const* name, int dim) {
    int varid = -1;
    check_nc(nc_def_var(fileid, name, NC_DOUBLE, 1, &dim, &varid));
    return varid;
  };
  int wave = define_1d("wavenumber", point);
  int lower = define_1d("wavenumber_lower", point);
  int upper = define_1d("wavenumber_upper", point);
  int weight = define_1d("quadrature_weight", point);
  int pres = define_1d("pressure", pressure);
  int temp_offset = define_1d("temperature_offset", offset);
  int temp_base = define_1d("nominal_temperature", pressure);
  int fraction = define_1d("reference_mole_fraction", species);
  int scatter = define_1d("scattering_coefficient", point);
  int linear_dims[4] = {linear, point, pressure, offset};
  int binary_dims[4] = {binary, point, pressure, offset};
  int moment_dims[2] = {point, moment};
  int klinear = -1, kbinary = -1, pmom = -1;
  check_nc(
      nc_def_var(fileid, "kappa_linear", NC_DOUBLE, 4, linear_dims, &klinear));
  check_nc(
      nc_def_var(fileid, "kappa_binary", NC_DOUBLE, 4, binary_dims, &kbinary));
  check_nc(
      nc_def_var(fileid, "phase_moment", NC_DOUBLE, 2, moment_dims, &pmom));
  put_text_attr(fileid, klinear, "units", "m2 mol-1");
  put_text_attr(fileid, kbinary, "units", "m5 mol-2");
  put_text_attr(fileid, scatter, "units", "m2 mol-1");
  check_nc(nc_enddef(fileid));

  double const wave_data[] = {100., 200.};
  double const lower_data[] = {99.5, 199.5};
  double const upper_data[] = {100.5, 200.5};
  double const weight_data[] = {1., 2.};
  double const pressure_data[] = {1.e5, 1.e6};
  double const offset_data[] = {-10., 10.};
  double const base_data[] = {300., 500.};
  double const fraction_data[] = {.75, .25};
  double const scattering_data[] = {1., 2.};
  double const linear_data[] = {2., 2., 2., 2., 3., 3., 3., 3.};
  double const binary_data[] = {.5, .5, .5, .5, .25, .25, .25, .25};
  double const moment_data[] = {1., .1, 1., .2};
  check_nc(nc_put_var_double(fileid, wave, wave_data));
  check_nc(nc_put_var_double(fileid, lower, lower_data));
  check_nc(nc_put_var_double(fileid, upper, upper_data));
  check_nc(nc_put_var_double(fileid, weight, weight_data));
  check_nc(nc_put_var_double(fileid, pres, pressure_data));
  check_nc(nc_put_var_double(fileid, temp_offset, offset_data));
  check_nc(nc_put_var_double(fileid, temp_base, base_data));
  check_nc(nc_put_var_double(fileid, fraction, fraction_data));
  check_nc(nc_put_var_double(fileid, scatter, scattering_data));
  check_nc(nc_put_var_double(fileid, klinear, linear_data));
  check_nc(nc_put_var_double(fileid, kbinary, binary_data));
  check_nc(nc_put_var_double(fileid, pmom, moment_data));
  check_nc(nc_close(fileid));
  return path;
}
#endif

TEST(TestOpacity, MoleculeLineAddsContinuumAndHandlesDimensionOrder) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto dataset = write_test_dataset();
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto op = harp::OpacityOptionsImpl::create();
  op->type("molecule-line").species_ids({0}).opacity_files({dataset.string()});
  harp::MoleculeLine line(op);

  auto conc = torch::zeros({1, 1, 3}, torch::kFloat64);
  conc[0][0][0] = 2.0;
  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::tensor({{1.0e5}}, torch::kFloat64);
  atm["temp"] = torch::tensor({{290.0}}, torch::kFloat64);
  atm["wavenumber"] = torch::tensor({20.0, 21.0, 22.0}, torch::kFloat64);

  auto result = line->forward(conc, atm).squeeze(-1).squeeze(-1).squeeze(-1);
  auto expected_sigma =
      torch::tensor({0.0, 3.1e-24, 5.1e-24}, torch::kFloat64) *
      (1.0e-4 * harp::constants::Avogadro);
  auto expected = expected_sigma * 2.0;
  EXPECT_TRUE(torch::allclose(result, expected, 1.0e-12, 1.0e-12));
  EXPECT_LT(result[0].item<double>(), 1.0e-250);
#endif
}

TEST(TestOpacity, MoleculeLineWeightsSplitWaterContinuumAtRuntime) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto dataset = write_test_dataset(true);
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto op = harp::OpacityOptionsImpl::create();
  op->type("molecule-line").species_ids({0}).opacity_files({dataset.string()});
  harp::MoleculeLine line(op);

  auto conc = torch::zeros({1, 1, 3}, torch::kFloat64);
  conc[0][0][0] = 1.0;
  conc[0][0][1] = 3.0;
  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::tensor({{1.0e5}}, torch::kFloat64);
  atm["temp"] = torch::tensor({{290.0}}, torch::kFloat64);
  atm["wavenumber"] = torch::tensor({20.0, 21.0, 22.0}, torch::kFloat64);

  auto result = line->forward(conc, atm).squeeze();
  // xH2O=0.25: continuum = 0.25*2e-24 + 0.75*6e-24 = 5e-24.
  // The legacy combined field is present but must be ignored when both split
  // fields exist.
  auto expected_sigma =
      torch::tensor({5.0e-24, 7.0e-24, 8.0e-24}, torch::kFloat64) *
      (1.0e-4 * harp::constants::Avogadro);
  EXPECT_TRUE(torch::allclose(result, expected_sigma, 1.0e-12, 1.0e-12));
#endif
}

TEST(TestOpacity, CIAHandlesBinaryPairsAndReversedSpeciesOrder) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto dataset = write_test_dataset();
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto op = harp::OpacityOptionsImpl::create();
  op->type("molecule-cia")
      .species_ids({2, 1})
      .opacity_files({dataset.string()});
  harp::MoleculeCIA cia(op);

  auto conc = torch::zeros({1, 1, 3}, torch::kFloat64);
  conc[0][0][1] = 3.0;
  conc[0][0][2] = 4.0;
  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::tensor({{1.0e5}}, torch::kFloat64);
  atm["temp"] = torch::tensor({{290.0}}, torch::kFloat64);
  atm["wavenumber"] = torch::tensor({20.0, 21.0, 22.0}, torch::kFloat64);

  auto result = cia->forward(conc, atm).squeeze(-1).squeeze(-1).squeeze(-1);
  auto expected_coeff =
      torch::tensor({0.0, 3.0e-46, 4.0e-46}, torch::kFloat64) *
      (1.0e-10 * harp::constants::Avogadro * harp::constants::Avogadro);
  auto expected = expected_coeff * 12.0;
  EXPECT_TRUE(torch::allclose(result, expected, 1.0e-12, 1.0e-12));
  EXPECT_LT(result[0].item<double>(), 1.0e-250);
#endif
}

TEST(TestOpacity, MoleculeTablesDistinguishCelsiusValuesFromIntervals) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto kelvin_dataset = write_test_dataset();
  auto celsius_dataset = write_test_dataset(false, true);
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto make_options = [](char const* type, std::vector<int> species_ids,
                         fs::path const& dataset) {
    auto options = harp::OpacityOptionsImpl::create();
    options->type(type)
        .species_ids(std::move(species_ids))
        .opacity_files({dataset.string()});
    return options;
  };

  harp::MoleculeLine line_k(make_options("molecule-line", {0}, kelvin_dataset));
  harp::MoleculeLine line_c(
      make_options("molecule-line", {0}, celsius_dataset));
  harp::MoleculeCIA cia_k(make_options("molecule-cia", {1, 2}, kelvin_dataset));
  harp::MoleculeCIA cia_c(
      make_options("molecule-cia", {1, 2}, celsius_dataset));

  auto expected_anomaly = torch::tensor({-10.0, 10.0}, torch::kFloat64);
  auto expected_temperature = torch::tensor({300.0, 500.0}, torch::kFloat64);
  for (auto const& anomaly :
       {line_k->temperature_anomaly, line_c->temperature_anomaly,
        cia_k->temperature_anomaly, cia_c->temperature_anomaly}) {
    EXPECT_TRUE(torch::allclose(anomaly, expected_anomaly));
  }
  for (auto const& log_temperature :
       {line_k->ln_temperature_base, line_c->ln_temperature_base,
        cia_k->ln_temperature_base, cia_c->ln_temperature_base}) {
    EXPECT_TRUE(torch::allclose(log_temperature.squeeze(-1).exp(),
                                expected_temperature));
  }

  auto unknown_dataset = write_test_dataset(false, false, "degF");
  EXPECT_THROW(
      harp::MoleculeLine(make_options("molecule-line", {0}, unknown_dataset)),
      c10::Error);
#endif
}

TEST(TestOpacity, MoleculeOpacitiesClampTemperatureAnomalyToTableBounds) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto dataset = write_test_dataset();
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto line_options = harp::OpacityOptionsImpl::create();
  line_options->type("molecule-line")
      .species_ids({0})
      .opacity_files({dataset.string()});
  harp::MoleculeLine line(line_options);

  auto cia_options = harp::OpacityOptionsImpl::create();
  cia_options->type("molecule-cia")
      .species_ids({1, 2})
      .opacity_files({dataset.string()});
  harp::MoleculeCIA cia(cia_options);

  auto conc = torch::ones({1, 1, 3}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::tensor({{1.0e5}}, torch::kFloat64);
  atm["wavenumber"] = torch::tensor({20.0, 21.0, 22.0}, torch::kFloat64);

  auto evaluate = [&](auto& opacity, double temperature) {
    atm["temp"] = torch::tensor({{temperature}}, torch::kFloat64);
    return opacity->forward(conc, atm);
  };

  auto line_lower_bound = evaluate(line, 290.0);
  auto line_below_bound = evaluate(line, 270.0);
  auto line_upper_bound = evaluate(line, 310.0);
  auto line_above_bound = evaluate(line, 330.0);
  EXPECT_TRUE(torch::allclose(line_below_bound, line_lower_bound));
  EXPECT_TRUE(torch::allclose(line_above_bound, line_upper_bound));

  auto cia_lower_bound = evaluate(cia, 290.0);
  auto cia_below_bound = evaluate(cia, 270.0);
  auto cia_upper_bound = evaluate(cia, 310.0);
  auto cia_above_bound = evaluate(cia, 330.0);
  EXPECT_TRUE(torch::allclose(cia_below_bound, cia_lower_bound));
  EXPECT_TRUE(torch::allclose(cia_above_bound, cia_upper_bound));
#endif
}

TEST(TestOpacity, RespqTableCombinesComponentsAndClampsBounds) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto dataset = write_respq_dataset();
  auto options = harp::OpacityOptionsImpl::create();
  options->type("respq-table")
      .species_ids({0, 1})
      .opacity_files({dataset.string()});
  harp::RespqTable table(options);

  auto conc = torch::tensor({{{3., 1.}}}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::tensor({{1.e5}}, torch::kFloat64);
  atm["temp"] = torch::tensor({{300.}}, torch::kFloat64);
  atm["wavenumber"] = torch::tensor({100., 200.}, torch::kFloat64);
  auto result = table->forward(conc, atm);
  EXPECT_TRUE(torch::allclose(result.select(-1, 0).flatten(),
                              torch::tensor({20., 24.}, torch::kFloat64)));
  EXPECT_TRUE(torch::allclose(result.select(-1, 1).flatten(),
                              torch::tensor({.2, 1. / 3.}, torch::kFloat64)));
  EXPECT_FALSE(torch::any(table->bounds_mask).item<bool>());

  atm["pres"] = torch::tensor({{1.e4}}, torch::kFloat64);
  atm["temp"] = torch::tensor({{1000.}}, torch::kFloat64);
  EXPECT_TRUE(torch::allclose(table->forward(conc, atm), result));
  EXPECT_TRUE(torch::all(table->bounds_mask).item<bool>());

  auto wrong = torch::tensor({{{2., 2.}}}, torch::kFloat64);
  EXPECT_THROW(table->forward(wrong, atm), c10::Error);

  auto band_options = harp::RadiationBandOptionsImpl::create();
  band_options->name("stellar")
      .solver_name("toon")
      .toon(harp::ToonMcKay89OptionsImpl::create())
      .nwave(2)
      .ncol(1)
      .nlyr(1);
  band_options->opacities()["respq"] = options;
  harp::RadiationBand band(band_options);
  EXPECT_EQ(band_options->weight(), (std::vector<double>{1., 2.}));
  EXPECT_EQ(band_options->toon()->wave_lower(),
            (std::vector<double>{99.5, 199.5}));
#endif
}

TEST(TestOpacity, MoleculeOpacitiesWarnOnceWhenQueriesAreClamped) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  auto dataset = write_test_dataset();
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto line_options = harp::OpacityOptionsImpl::create();
  line_options->type("molecule-line")
      .species_ids({0})
      .opacity_files({dataset.string()})
      .warn_out_of_bounds(true);
  harp::MoleculeLine line(line_options);

  auto cia_options = harp::OpacityOptionsImpl::create();
  cia_options->type("molecule-cia")
      .species_ids({1, 2})
      .opacity_files({dataset.string()})
      .warn_out_of_bounds(true);
  harp::MoleculeCIA cia(cia_options);

  auto conc = torch::ones({1, 1, 3}, torch::kFloat64);
  std::map<std::string, torch::Tensor> atm;
  atm["pres"] = torch::tensor({{5.0e4}}, torch::kFloat64);
  atm["temp"] = torch::tensor({{250.0}}, torch::kFloat64);
  atm["wavenumber"] = torch::tensor({19.0, 21.0, 23.0}, torch::kFloat64);

  testing::internal::CaptureStderr();
  line->forward(conc, atm);
  line->forward(conc, atm);
  auto const line_warning = testing::internal::GetCapturedStderr();
  EXPECT_NE(line_warning.find("MoleculeLine wavenumber query range"),
            std::string::npos);
  EXPECT_EQ(line_warning.find("MoleculeLine wavenumber query range"),
            line_warning.rfind("MoleculeLine wavenumber query range"));
  EXPECT_NE(line_warning.find("MoleculeLine pressure query range"),
            std::string::npos);
  EXPECT_EQ(line_warning.find("MoleculeLine pressure query range"),
            line_warning.rfind("MoleculeLine pressure query range"));
  EXPECT_NE(line_warning.find("MoleculeLine temperature anomaly query range"),
            std::string::npos);
  EXPECT_EQ(line_warning.find("MoleculeLine temperature anomaly query range"),
            line_warning.rfind("MoleculeLine temperature anomaly query range"));

  testing::internal::CaptureStderr();
  cia->forward(conc, atm);
  cia->forward(conc, atm);
  auto const cia_warning = testing::internal::GetCapturedStderr();
  EXPECT_NE(cia_warning.find("MoleculeCIA wavenumber query range"),
            std::string::npos);
  EXPECT_EQ(cia_warning.find("MoleculeCIA wavenumber query range"),
            cia_warning.rfind("MoleculeCIA wavenumber query range"));
  EXPECT_NE(cia_warning.find("MoleculeCIA pressure query range"),
            std::string::npos);
  EXPECT_EQ(cia_warning.find("MoleculeCIA pressure query range"),
            cia_warning.rfind("MoleculeCIA pressure query range"));
  EXPECT_NE(cia_warning.find("MoleculeCIA temperature anomaly query range"),
            std::string::npos);
  EXPECT_EQ(cia_warning.find("MoleculeCIA temperature anomaly query range"),
            cia_warning.rfind("MoleculeCIA temperature anomaly query range"));
#endif
}

TEST(TestOpacity, NewOpacityTypesParseFromYaml) {
  harp::species_names = {"H2O", "H2", "He"};
  harp::species_weights = {18.0e-3, 2.0e-3, 4.0e-3};

  auto yaml_path = fs::temp_directory_path() / "pyharp_test_opacity.yaml";
  std::ofstream out(yaml_path);
  out << "opacities:\n"
      << "  line:\n"
      << "    type: molecule-line\n"
      << "    data: [/tmp/mock.nc]\n"
      << "    species: [H2O]\n"
      << "    warn_out_of_bounds: true\n"
      << "  cia_pair:\n"
      << "    type: molecule-cia\n"
      << "    data: [/tmp/mock.nc]\n"
      << "    species: [H2, He]\n";
  out.close();

  auto line = harp::OpacityOptionsImpl::from_yaml(yaml_path.string(), "line");
  EXPECT_EQ(line->type(), "molecule-line");
  ASSERT_EQ(line->species_ids().size(), 1);
  EXPECT_EQ(line->species_ids()[0], 0);
  EXPECT_TRUE(line->warn_out_of_bounds());

  auto cia =
      harp::OpacityOptionsImpl::from_yaml(yaml_path.string(), "cia_pair");
  EXPECT_EQ(cia->type(), "molecule-cia");
  ASSERT_EQ(cia->species_ids().size(), 2);
  EXPECT_EQ(cia->species_ids()[0], 1);
  EXPECT_EQ(cia->species_ids()[1], 2);
  EXPECT_FALSE(cia->warn_out_of_bounds());
}

#ifdef NETCDFOUTPUT
// A smooth, strictly positive self-CIA table for H2O on 31 wavenumbers,
// 9 pressures and 9 temperature anomalies. No query below sits on a node.
// ln(sigma) changes by 0.05 across a temperature-anomaly bracket, small next
// to |ln(sigma)| ~ 17, so a float32 gradient that reduces over wavenumber
// before the bracket difference is taken loses accuracy.
fs::path write_cia_gradient_dataset() {
  auto path = fs::temp_directory_path() / "pyharp_test_cia_gradient.nc";
  int const nwave = 31, npres = 9, ntemp = 9;

  int fileid = -1;
  check_nc(nc_create(path.c_str(), NC_CLOBBER, &fileid));

  int dim_wave = -1, dim_pres = -1, dim_temp = -1;
  check_nc(nc_def_dim(fileid, "wavenumber", nwave, &dim_wave));
  check_nc(nc_def_dim(fileid, "pressure", npres, &dim_pres));
  check_nc(nc_def_dim(fileid, "del_temperature", ntemp, &dim_temp));

  int var_wave = -1, var_pres = -1, var_del_temp = -1, var_temp = -1;
  int var_cia = -1;
  check_nc(
      nc_def_var(fileid, "wavenumber", NC_DOUBLE, 1, &dim_wave, &var_wave));
  check_nc(nc_def_var(fileid, "pressure", NC_DOUBLE, 1, &dim_pres, &var_pres));
  check_nc(nc_def_var(fileid, "del_temperature", NC_DOUBLE, 1, &dim_temp,
                      &var_del_temp));
  check_nc(
      nc_def_var(fileid, "temperature", NC_DOUBLE, 1, &dim_pres, &var_temp));
  int const dims3[] = {dim_wave, dim_pres, dim_temp};
  check_nc(nc_def_var(fileid, "binary_absorption_coefficient_h2o_h2o",
                      NC_DOUBLE, 3, dims3, &var_cia));

  put_text_attr(fileid, var_wave, "units", "cm^-1");
  put_text_attr(fileid, var_pres, "units", "Pa");
  put_text_attr(fileid, var_del_temp, "units", "K");
  put_text_attr(fileid, var_temp, "units", "K");
  put_text_attr(fileid, var_cia, "units", "cm^5 molecule^-2");
  check_nc(nc_enddef(fileid));

  std::vector<double> wave(nwave), pres(npres), temp(npres), del_temp(ntemp);
  for (int i = 0; i < nwave; ++i) wave[i] = 100.0 + 30.0 * i;
  for (int i = 0; i < npres; ++i) {
    pres[i] = 1.0e4 * std::pow(10.0, 0.25 * i);
    temp[i] = 250.0 + 10.0 * i;
  }
  for (int i = 0; i < ntemp; ++i) del_temp[i] = -100.0 + 25.0 * i;

  std::vector<double> sigma(nwave * npres * ntemp);
  for (int iw = 0; iw < nwave; ++iw) {
    double const shape = std::exp(-std::pow((wave[iw] - 550.0) / 350.0, 2));
    for (int ip = 0; ip < npres; ++ip) {
      for (int it = 0; it < ntemp; ++it) {
        sigma[(iw * npres + ip) * ntemp + it] = 1.0e-45 * shape *
                                                (1.0 + 0.0625 * ip) *
                                                std::exp(0.002 * del_temp[it]);
      }
    }
  }

  check_nc(nc_put_var_double(fileid, var_wave, wave.data()));
  check_nc(nc_put_var_double(fileid, var_pres, pres.data()));
  check_nc(nc_put_var_double(fileid, var_del_temp, del_temp.data()));
  check_nc(nc_put_var_double(fileid, var_temp, temp.data()));
  check_nc(nc_put_var_double(fileid, var_cia, sigma.data()));
  check_nc(nc_close(fileid));
  return path;
}

// Temperature gradient of the summed float32 CIA opacity on 64 wavenumbers x
// 8 columns x 32 layers, against the same inputs in float64. Every one of the
// 256 entries must satisfy |reference - candidate| <= 1e-14 + 3e-5 |candidate|.
void expect_cia_float32_temperature_gradient(torch::Device device,
                                             bool varying) {
  if (device.is_cuda() && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto dataset = write_cia_gradient_dataset();
  harp::species_names = {"H2O"};
  harp::species_weights = {18.0e-3};

  auto options = harp::OpacityOptionsImpl::create();
  options->type("molecule-cia")
      .species_ids({0})
      .opacity_files({dataset.string()});

  // Inputs are made in float64 and rounded once to float32, so that the
  // reference sees exactly the float32 values.
  auto f64 = torch::TensorOptions().dtype(torch::kFloat64);
  auto conc = torch::linspace(0.1, 1.0, 256, f64).view({8, 32, 1});
  auto wavenumber = torch::linspace(150.0, 950.0, 64, f64);
  torch::Tensor pres, temp;
  if (varying) {
    pres = torch::logspace(std::log10(2.0e4), std::log10(8.0e5), 256, 10.0, f64)
               .view({8, 32});
    temp = torch::linspace(270.0, 340.0, 256, f64).view({8, 32});
  } else {
    pres = torch::full({8, 32}, 3.0e4, f64);
    temp = torch::full({8, 32}, 305.0, f64);
  }

  auto temperature_gradient = [&](torch::Dtype dtype) {
    harp::MoleculeCIA cia(options);
    cia->to(device, dtype);
    auto cast = [&](torch::Tensor const& x) {
      return x.to(torch::kFloat32).to(device, dtype);
    };
    auto conc_d = cast(conc).requires_grad_(true);
    auto temp_d = cast(temp).requires_grad_(true);
    std::map<std::string, torch::Tensor> atm;
    atm["pres"] = cast(pres).requires_grad_(true);
    atm["temp"] = temp_d;
    atm["wavenumber"] = cast(wavenumber);
    auto out = cia->forward(conc_d, atm);
    EXPECT_TRUE(torch::all(torch::isfinite(out)).item<bool>());
    out.sum().backward();
    return temp_d.grad().to(torch::kCPU, torch::kFloat64);
  };

  auto reference = temperature_gradient(torch::kFloat64);
  auto candidate = temperature_gradient(torch::kFloat32);

  auto bound = 1.0e-14 + 3.0e-5 * candidate.abs();
  auto failed = (reference - candidate).abs() > bound;
  auto relative = ((reference - candidate) / reference).abs();
  ::testing::Test::RecordProperty("exceedances", failed.sum().item<int64_t>());
  std::ostringstream max_relative;
  max_relative << std::scientific << relative.max().item<double>();
  ::testing::Test::RecordProperty("max_relative_difference",
                                  max_relative.str());
  EXPECT_EQ(failed.sum().item<int64_t>(), 0)
      << failed.sum().item<int64_t>() << "/256 entries exceed the bound; max "
      << "relative difference " << relative.max().item<double>();
}
#endif

TEST(TestOpacity, CIAFloat32TemperatureGradientBroadcastCPU) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_cia_float32_temperature_gradient(torch::kCPU, false);
#endif
}

TEST(TestOpacity, CIAFloat32TemperatureGradientVaryingCPU) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_cia_float32_temperature_gradient(torch::kCPU, true);
#endif
}

TEST(TestOpacity, CIAFloat32TemperatureGradientBroadcastCUDA) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_cia_float32_temperature_gradient(torch::kCUDA, false);
#endif
}

TEST(TestOpacity, CIAFloat32TemperatureGradientVaryingCUDA) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_cia_float32_temperature_gradient(torch::kCUDA, true);
#endif
}

#ifdef NETCDFOUTPUT
// numpy.linspace and numpy.geomspace in float64, operation for operation, so
// that the float32 inputs below are the bytes of the original audit fixture.
std::vector<double> numpy_linspace(double start, double stop, int num) {
  std::vector<double> out(num);
  double const step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) out[i] = i * step + start;
  out[num - 1] = stop;
  return out;
}

std::vector<double> numpy_geomspace(double start, double stop, int num) {
  auto out = numpy_linspace(std::log10(start), std::log10(stop), num);
  for (auto& x : out) x = std::pow(10.0, x);
  out[0] = start;
  out[num - 1] = stop;
  return out;
}

// The H2O-H2O table of the post-#165 audit: 32 wavenumbers, 8 pressures, 9
// temperature anomalies, base temperature 300 K and
// sigma = exp(-20 + a_wave + b_pres + c_temp) m^5 mol^-2.
fs::path write_audit_cia_dataset() {
  auto path = fs::temp_directory_path() / "pyharp_test_cia_audit.nc";
  auto wave = numpy_linspace(100.0, 1000.0, 32);
  auto pres = numpy_geomspace(1.0e4, 1.0e6, 8);
  auto del_temp = numpy_linspace(-100.0, 100.0, 9);
  std::vector<double> temp(8, 300.0);
  auto a = numpy_linspace(0.0, 1.0, 32);
  auto b = numpy_linspace(0.0, 0.3, 8);
  auto c = numpy_linspace(-0.2, 0.2, 9);
  std::vector<double> sigma(32 * 8 * 9);
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int k = 0; k < 9; ++k) {
        sigma[(i * 8 + j) * 9 + k] = std::exp(-20.0 + a[i] + b[j] + c[k]);
      }
    }
  }

  int fileid = -1;
  check_nc(nc_create(path.c_str(), NC_CLOBBER, &fileid));
  int dims[3], var_wave = -1, var_pres = -1, var_del_temp = -1;
  int var_temp = -1, var_cia = -1;
  check_nc(nc_def_dim(fileid, "wavenumber", 32, &dims[0]));
  check_nc(nc_def_var(fileid, "wavenumber", NC_DOUBLE, 1, &dims[0], &var_wave));
  put_text_attr(fileid, var_wave, "units", "cm^-1");
  check_nc(nc_def_dim(fileid, "pressure", 8, &dims[1]));
  check_nc(nc_def_var(fileid, "pressure", NC_DOUBLE, 1, &dims[1], &var_pres));
  put_text_attr(fileid, var_pres, "units", "Pa");
  check_nc(nc_def_dim(fileid, "del_temperature", 9, &dims[2]));
  check_nc(nc_def_var(fileid, "del_temperature", NC_DOUBLE, 1, &dims[2],
                      &var_del_temp));
  put_text_attr(fileid, var_del_temp, "units", "K");
  check_nc(
      nc_def_var(fileid, "temperature", NC_DOUBLE, 1, &dims[1], &var_temp));
  put_text_attr(fileid, var_temp, "units", "K");
  check_nc(nc_def_var(fileid, "binary_absorption_coefficient_h2o_h2o",
                      NC_DOUBLE, 3, dims, &var_cia));
  put_text_attr(fileid, var_cia, "units", "m^5 mol^-2");
  check_nc(nc_enddef(fileid));
  check_nc(nc_put_var_double(fileid, var_wave, wave.data()));
  check_nc(nc_put_var_double(fileid, var_pres, pres.data()));
  check_nc(nc_put_var_double(fileid, var_del_temp, del_temp.data()));
  check_nc(nc_put_var_double(fileid, var_temp, temp.data()));
  check_nc(nc_put_var_double(fileid, var_cia, sigma.data()));
  check_nc(nc_close(fileid));
  return path;
}

// The four float32 temperature-gradient gates of the post-#165 audit: the
// summed CIA opacity on 64 wavenumbers x 8 columns x 32 layers, one thread,
// compared entry by entry with the graph that expands all three queries to
// (nwave, ncol, nlyr), as MoleculeCIA did before #165. Each of the 256
// entries must satisfy |reference - candidate| <= 1e-14 + 3e-5 |candidate|.
void expect_audit_cia_temperature_gradient(torch::Device device, bool varying) {
  if (device.is_cuda() && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto dataset = write_audit_cia_dataset();
  harp::species_names = {"H2O"};
  harp::species_weights = {18.0e-3};

  int const threads = torch::get_num_threads();
  torch::set_num_threads(1);

  auto options = harp::OpacityOptionsImpl::create();
  options->type("molecule-cia")
      .species_ids({0})
      .opacity_files({dataset.string()})
      .warn_out_of_bounds(false);
  harp::MoleculeCIA cia(options);
  cia->to(device, torch::kFloat32);

  auto f32 = [&](std::vector<double> const& x) {
    return torch::tensor(x, torch::kFloat64).to(torch::kFloat32).to(device);
  };
  auto conc = f32(numpy_linspace(0.1, 1.0, 256)).view({8, 32, 1});
  auto wave = f32(numpy_linspace(150.0, 950.0, 64));
  torch::Tensor pres, temp;
  if (varying) {
    pres = f32(numpy_geomspace(2.0e4, 8.0e5, 256)).view({8, 32});
    temp = f32(numpy_linspace(270.0, 340.0, 256)).view({8, 32});
  } else {
    // Zero-stride inputs; each of the 256 entries is its own gradient.
    pres = f32({3.0e4}).expand({8, 32});
    temp = f32({305.0}).expand({8, 32});
  }

  auto expanded_forward = [&](torch::Tensor const& conc_d,
                              torch::Tensor const& pres_d,
                              torch::Tensor const& temp_d) {
    auto lnp = pres_d.log();
    auto temperature_base =
        harp::interpn({lnp}, {cia->ln_pressure}, cia->ln_temperature_base)
            .squeeze(-1)
            .exp();
    auto del_temp = temp_d - temperature_base;
    auto coeff =
        harp::interpn(
            {wave.unsqueeze(-1).unsqueeze(-1).expand({64, 8, 32}),
             lnp.unsqueeze(0).expand({64, 8, 32}),
             del_temp.unsqueeze(0).expand({64, 8, 32})},
            {cia->wavenumber, cia->ln_pressure, cia->temperature_anomaly},
            cia->ln_sigma_binary, false)
            .exp();
    auto species = conc_d.select(-1, 0).unsqueeze(0).unsqueeze(-1);
    return coeff * species * species;
  };

  auto temperature_gradient = [&](bool reference) {
    auto conc_d = conc.clone().requires_grad_(true);
    auto pres_d = pres.detach().requires_grad_(true);
    auto temp_d = temp.detach().requires_grad_(true);
    torch::Tensor out;
    if (reference) {
      out = expanded_forward(conc_d, pres_d, temp_d);
    } else {
      std::map<std::string, torch::Tensor> atm;
      atm["pres"] = pres_d;
      atm["temp"] = temp_d;
      atm["wavenumber"] = wave;
      out = cia->forward(conc_d, atm);
    }
    EXPECT_TRUE(torch::all(torch::isfinite(out)).item<bool>());
    auto grads = torch::autograd::grad({out.sum()}, {conc_d, pres_d, temp_d});
    return grads[2].to(torch::kCPU, torch::kFloat64);
  };

  auto reference = temperature_gradient(true);
  auto candidate = temperature_gradient(false);
  torch::set_num_threads(threads);

  auto bound = 1.0e-14 + 3.0e-5 * candidate.abs();
  auto failed = (reference - candidate).abs() > bound;
  ::testing::Test::RecordProperty("exceedances", failed.sum().item<int64_t>());
  EXPECT_EQ(failed.sum().item<int64_t>(), 0)
      << failed.sum().item<int64_t>() << "/256 entries exceed the bound at "
      << failed.nonzero();
}
#endif

TEST(TestOpacity, CIAAuditTemperatureGradientBroadcastCPU) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_audit_cia_temperature_gradient(torch::kCPU, false);
#endif
}

TEST(TestOpacity, CIAAuditTemperatureGradientVaryingCPU) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_audit_cia_temperature_gradient(torch::kCPU, true);
#endif
}

TEST(TestOpacity, CIAAuditTemperatureGradientBroadcastCUDA) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_audit_cia_temperature_gradient(torch::kCUDA, false);
#endif
}

TEST(TestOpacity, CIAAuditTemperatureGradientVaryingCUDA) {
#ifndef NETCDFOUTPUT
  GTEST_SKIP() << "NetCDF support is disabled";
#else
  expect_audit_cia_temperature_gradient(torch::kCUDA, true);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
