// harp
#include "disort_options_flux.hpp"

namespace harp {

disort::DisortOptions disort_flux_sw(int nwave, int ncol, int nlyr, int nstr) {
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

disort::DisortOptions disort_flux_lw(double wmin, double wmax, int nwave,
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
