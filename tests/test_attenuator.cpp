// external
#include <gtest/gtest.h>

// opacity
#include <opacity/h2so4_simple.hpp>
#include <opacity/s8_fuller.hpp>

using namespace harp;

TEST(TestOpacity, s8_fuller) {
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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
