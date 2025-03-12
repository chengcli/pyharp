// harp
#include "disort_config.hpp"

namespace harp {

disort::DisortOptions disort_config(int nwave, int ncol, int nlyr, int nstr) {
  disort::DisortOptions op;

  op.nwave(nwave);
  op.ncol(ncol);

  op.ds().nlyr = nlyr;
  op.ds().nstr = nstr;
  op.ds().nmom = nstr;

  return op;
}

disort::DisortOptions disort_config_sw(int nwave, int ncol, int nlyr,
                                       int nstr) {
  disort::DisortOptions op;

  op.header("running disort shortwave");
  op.flags(
      "lamber,quiet,onlyfl,"
      "intensity_correction,old_intensity_correction");

  op.nwave(nwave);
  op.ncol(ncol);

  op.ds().nlyr = nlyr;
  op.ds().nstr = nstr;
  op.ds().nmom = nstr;

  return op;
}

disort::DisortOptions disort_config_lw(double wmin, double wmax, int nwave,
                                       int ncol, int nlyr, int nstr) {
  disort::DisortOptions op;

  op.header("running disort longwave");
  op.flags(
      "lamber,quiet,onlyfl,planck,"
      "intensity_correction,old_intensity_correction");

  op.nwave(nwave);
  op.ncol(ncol);
  op.wave_lower(std::vector<double>(nwave, wmin));
  op.wave_upper(std::vector<double>(nwave, wmax));

  op.ds().nlyr = nlyr;
  op.ds().nstr = nstr;
  op.ds().nmom = nstr;

  return op;
}

}  // namespace harp
