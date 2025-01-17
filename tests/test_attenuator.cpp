// external
#include <gtest/gtest.h>

// opacity
#include <opacity/h2so4_simple.hpp>
#include <opacity/s8_fuller.hpp>

using namespace harp;

TEST(TestOpacity, s8_fuller) {
  S8RTOptions op1;
  op1.species_id(0);

  H2SO4RTOptions op2;
  op2.species_id(1);

  S8Fuller s8(op1);
  H2SO4Simple h2so4(op2);

  // std::cout << "h2so4 wave = " << h2so4->kwave << std::endl;
  // std::cout << "h2so4 data = " << h2so4->kdata << std::endl;

  int ncol = 1;
  int nlyr = 1;
  int nspecies = 2;
  auto conc = torch::ones({ncol, nlyr, nspecies}, torch::kFloat64);
  auto result1 = s8->forward(s8->kwave, conc);
  auto result2 = h2so4->forward(s8->kwave, conc);
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
