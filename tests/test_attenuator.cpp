// C/C++
#include <cstring>
#include <filesystem>
#include <fstream>

// base
#include <configure.h>

// external
#include <gtest/gtest.h>

// harp
#include <harp/constants.h>

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

fs::path write_test_dataset(bool split_continuum = false) {
  auto path = fs::temp_directory_path() /
              (split_continuum ? "pyharp_test_molecule_line_split.nc"
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
  put_text_attr(fileid, var_del_temp, "units", "K");
  put_text_attr(fileid, var_temperature, "units", "K");
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
  double const temperature[] = {300.0, 500.0};

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
      << "  cia_pair:\n"
      << "    type: molecule-cia\n"
      << "    data: [/tmp/mock.nc]\n"
      << "    species: [H2, He]\n";
  out.close();

  auto line = harp::OpacityOptionsImpl::from_yaml(yaml_path.string(), "line");
  EXPECT_EQ(line->type(), "molecule-line");
  ASSERT_EQ(line->species_ids().size(), 1);
  EXPECT_EQ(line->species_ids()[0], 0);

  auto cia =
      harp::OpacityOptionsImpl::from_yaml(yaml_path.string(), "cia_pair");
  EXPECT_EQ(cia->type(), "molecule-cia");
  ASSERT_EQ(cia->species_ids().size(), 2);
  EXPECT_EQ(cia->species_ids()[0], 1);
  EXPECT_EQ(cia->species_ids()[1], 2);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
