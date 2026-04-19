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

fs::path write_test_dataset() {
  auto path = fs::temp_directory_path() / "pyharp_test_molecule_line.nc";
  int fileid = -1;
  check_nc(nc_create(path.c_str(), NC_CLOBBER, &fileid));

  int dim_del_temp = -1, dim_pressure = -1, dim_wavenumber = -1;
  check_nc(nc_def_dim(fileid, "del_temperature", 2, &dim_del_temp));
  check_nc(nc_def_dim(fileid, "pressure", 2, &dim_pressure));
  check_nc(nc_def_dim(fileid, "wavenumber", 3, &dim_wavenumber));

  int var_wavenumber = -1, var_pressure = -1, var_del_temp = -1;
  int var_temperature = -1, var_line = -1, var_cont = -1, var_cia = -1;
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
  check_nc(nc_def_var(fileid, "binary_absorption_coefficient_h2_he", NC_DOUBLE,
                      3, dims3, &var_cia));

  put_text_attr(fileid, var_wavenumber, "units", "cm^-1");
  put_text_attr(fileid, var_pressure, "units", "Pa");
  put_text_attr(fileid, var_del_temp, "units", "K");
  put_text_attr(fileid, var_temperature, "units", "K");
  put_text_attr(fileid, var_line, "units", "cm^2 molecule^-1");
  put_text_attr(fileid, var_cont, "units", "cm^2 molecule^-1");
  put_text_attr(fileid, var_cia, "units", "cm^5 molecule^-2");

  check_nc(nc_enddef(fileid));

  double const wavenumber[] = {20.0, 21.0, 22.0};
  double const pressure[] = {1.0e5, 1.0e6};
  double const del_temp[] = {-10.0, 10.0};
  double const temperature[] = {300.0, 500.0};

  std::vector<double> sigma_line(2 * 2 * 3);
  std::vector<double> sigma_cont(2 * 2 * 3);
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

  check_nc(nc_put_var_double(fileid, var_wavenumber, wavenumber));
  check_nc(nc_put_var_double(fileid, var_pressure, pressure));
  check_nc(nc_put_var_double(fileid, var_del_temp, del_temp));
  check_nc(nc_put_var_double(fileid, var_temperature, temperature));
  check_nc(nc_put_var_double(fileid, var_line, sigma_line.data()));
  check_nc(nc_put_var_double(fileid, var_cont, sigma_cont.data()));
  check_nc(nc_put_var_double(fileid, var_cia, sigma_cia.data()));
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
      torch::tensor({1.1e-24, 3.1e-24, 5.1e-24}, torch::kFloat64) *
      (1.0e-4 * harp::constants::Avogadro);
  auto expected = expected_sigma * 2.0;
  EXPECT_TRUE(torch::allclose(result, expected, 1.0e-12, 1.0e-12));
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
      torch::tensor({2.0e-46, 3.0e-46, 4.0e-46}, torch::kFloat64) *
      (1.0e-10 * harp::constants::Avogadro * harp::constants::Avogadro);
  auto expected = expected_coeff * 12.0;
  EXPECT_TRUE(torch::allclose(result, expected, 1.0e-12, 1.0e-12));
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
