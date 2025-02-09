// external
#include <gtest/gtest.h>

// opacity
#include <opacity/h2so4_simple.hpp>
#include <opacity/rfm.hpp>
#include <opacity/s8_fuller.hpp>

using namespace harp;

TEST(TestOpacity, sw) {
  AttenuatorOptions op;
  op.species_names({"S8", "H2SO4"});
  op.species_weights({256.0e-3, 98.0e-3});

  op.species_ids({0}).opacity_files({"s8_k_fuller.txt"});
  S8Fuller s8(op);

  op.species_ids({1}).opacity_files({"h2so4.txt"});
  H2SO4Simple h2so4(op);

  // std::cout << "h2so4 wave = " << h2so4->kwave << std::endl;
  // std::cout << "h2so4 data = " << h2so4->kdata << std::endl;

  int ncol = 1;
  int nlyr = 1;
  int nspecies = 2;
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  std::map<std::string, torch::Tensor> kwargs;
  kwargs["wavelength"] = s8->kwave;

  auto result1 = s8->forward(conc, kwargs);
  auto result2 = h2so4->forward(conc, kwargs);
  // std::cout << "result1 = " << result1 << std::endl;
  // std::cout << "result2 = " << result2 << std::endl;
  // std::cout << result2.sizes() << std::endl;

  // attenuation [1/m]
  auto result = result1 + result2;

  // single scattering albedo
  result.select(-1, 1) = result.select(-1, 1) / result.select(-1, 0);

  std::cout << "result = " << result << std::endl;
};

TEST(TestOpacity, lw) {
  AttenuatorOptions op;
  op.species_names({"CO2", "H2O"});
  op.species_weights({44.0e-3, 18.0e-3});

  op.species_ids({0}).opacity_files({"amarsw-ck-B1.nc"});
  RFM co2(op);

  std::cout << "co2 = " << co2->kdata << std::endl;
  std::cout << "pres = " << exp(co2->krefatm[RFMImpl::IPR]) << std::endl;
  std::cout << "temp = " << co2->krefatm[RFMImpl::ITM] << std::endl;
  std::cout << "kaxis = " << co2->kaxis << std::endl;

  op.species_ids({1}).opacity_files({"amarsw-ck-B1.nc"});
  RFM h2o(op);
}

TEST(TestOpacity, get_reftemp) {
  AttenuatorOptions op;
  op.species_names({"CO2", "H2O"});
  op.species_weights({44.0e-3, 18.0e-3});

  op.species_ids({0}).opacity_files({"amarsw-ck-B1.nc"});
  RFM co2(op);

  auto coord = (torch::ones({2, 2}, torch::kFloat64) * 100.e5).log();
  std::cout << "coord = " << coord << std::endl;

  auto temp = get_reftemp(coord, co2->krefatm[RFMImpl::IPR],
                          co2->krefatm[RFMImpl::ITM]);
  std::cout << "temp = " << temp << std::endl;
}

TEST(TestOpacity, forward) {
  AttenuatorOptions op;
  op.species_names({"CO2", "H2O"});
  op.species_weights({44.0e-3, 18.0e-3});

  op.species_ids({0}).opacity_files({"amarsw-ck-B1.nc"});
  RFM co2(op);

  op.species_ids({1}).opacity_files({"amarsw-ck-B1.nc"});
  RFM h2o(op);

  int ncol = 1;
  int nlyr = 1;
  int nspecies = 2;
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  std::map<std::string, torch::Tensor> kwargs;
  kwargs["pres"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 10.e5;
  kwargs["temp"] = torch::ones({ncol, nlyr}, torch::kFloat64) * 300.0;

  auto result1 = co2->forward(conc, kwargs);
  auto result2 = h2o->forward(conc, kwargs);

  std::cout << "result1 = " << result1 << std::endl;
  std::cout << "result2 = " << result2.squeeze(1).squeeze(1) << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
