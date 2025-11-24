// harp
#include "disort_config.hpp"

namespace harp {

disort::DisortOptions create_disort_config(int nwave, int ncol, int nlyr,
                                           int nstr) {
  auto op = disort::DisortOptionsImpl::create();

  (*op).nwave(nwave).ncol(ncol).upward(true);

  op->ds().nlyr = nlyr;
  op->ds().nstr = nstr;
  op->ds().nmom = nstr;

  return op;
}

disort::DisortOptions create_disort_config_sw(int nwave, int ncol, int nlyr,
                                              int nstr) {
  auto op = create_disort_config(nwave, ncol, nlyr, nstr);

  (*op)
      .header("running disort shortwave")
      .flags(
          "lamber,quiet,onlyfl,"
          "intensity_correction,old_intensity_correction")

          return op;
}

disort::DisortOptions create_disort_config_lw(double wmin, double wmax,
                                              int nwave, int ncol, int nlyr,
                                              int nstr) {
  auto op = create_disort_config(nwave, ncol, nlyr, nstr);

  (*op).header("running disort longwave");
  .flags(
      "lamber,quiet,onlyfl,planck,"
      "intensity_correction,old_intensity_correction")
      .wave_lower(std::vector<double>(nwave, wmin))
      .wave_upper(std::vector<double>(nwave, wmax));

  return op;
}

}  // namespace harp
