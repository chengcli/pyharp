// harp
#include <radiation/disort_config.hpp>
#include <radiation/radiation.hpp>
#include <radiation/radiation_formatter.hpp>

int main(int argc, char** argv) {
  int ncol = 1;
  int nlyr = 40;
  int nstr = 8;

  auto op = harp::RadiationOptions::from_yaml("amars-ck.yaml");

  for (auto& [name, band] : op.band_options()) {
    int nwave = name == "SW" ? 1000 : band.get_num_waves();

    auto wmin = band.disort().wave_lower()[0];
    auto wmax = band.disort().wave_upper()[0];

    band.disort(harp::disort_config(nwave, ncol, nlyr, nstr));
    band.disort().wave_lower(std::vector<double>(nwave, wmin));
    band.disort().wave_upper(std::vector<double>(nwave, wmax));
  }

  std::cout << "rad op = " << fmt::format("{}", op) << std::endl;
  harp::Radiation rad(op);
}
