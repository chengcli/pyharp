// harp
#include <harp/rtsolver/toon_mckay89.hpp>

int main(int argc, char** argv) {
  auto op = harp::ToonMcKay89OptionsImpl::create();
  op->wave_lower({200., 500., 1000.});
  op->wave_upper({500., 1000., 2000.});

  op->report(std::cout);
  harp::ToonMcKay89 toon(op);

  int nwave = op->wave_lower().size();
  int nlyr = 10;
  int ncol = 2;
  int nprop = 3;

  auto prop = 0.5 * torch::ones({nwave, ncol, nlyr, nprop}, torch::kFloat64);
  prop.select(-1, 0) = 0.1;
  prop.select(-1, 1) = 0.2;
  prop.select(-1, 2) = 0.3;
  std::map<std::string, torch::Tensor> bc;
  bc["fbeam"] = torch::ones({nwave, ncol}, torch::kFloat64);
  bc["umu0"] = torch::ones({ncol}, torch::kFloat64) * 0.2;
  bc["albedo"] = torch::ones({nwave, ncol}, torch::kFloat64) * 0.3;

  auto sw_flx = toon(prop, &bc);

  std::cout << "sw_flx_up = " << sw_flx.select(-1, 0) << "\n";
  std::cout << "sw_flx_dn = " << sw_flx.select(-1, 1) << "\n";

  auto temf = torch::ones({ncol, nlyr + 1}, torch::kFloat64) * 300.0;
  auto lw_flx = toon(prop, &bc, "", temf);

  std::cout << "lw_flx_up = " << lw_flx.select(-1, 0) << "\n";
  std::cout << "lw_flx_dn = " << lw_flx.select(-1, 1) << "\n";
}
