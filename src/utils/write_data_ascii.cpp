// C/C++
#include <fstream>
#include <iostream>

// harp
#include "write_bin_ascii.hpp"

namespace harp {
void write_bin_ascii_header(RadiationBand const &band, std::string fname) {
  FILE *pfile = fopen(fname.c_str(), "w");

  fprintf(pfile, "# Bin Radiances of Band %s: %.3g - %.3g\n",
          band->name().c_str(), band->options.wmin(), band->options.wmax());
  ;
  auto const &rayOutput = band->rayOutput;
  fprintf(pfile, "# Ray output size: %lu\n", rayOutput.size(0));

  fprintf(pfile, "# Polar angles: ");
  for (int i = 0; i < rayOutput.size(0); ++i) {
    fprintf(pfile, "%.3f", rad2deg(acos(rayOutput[i][0].item().toFloat())));
  }
  fprintf(pfile, "\n");

  fprintf(pfile, "# Azimuthal angles: ");
  for (int i = 0; i < rayOutput.size(0); ++i) {
    fprintf(pfile, "%.3f", rad2deg(rayOutput[i][1].item().toFloat()));
  }
  fprintf(pfile, "\n");

  fprintf(pfile, "#%12s%12s", "Wave", "Weight");
  for (int i = 0; i < rayOutput.size(0); ++i) {
    fprintf(pfile, "%12s%lu", "Radiance", j + 1);
  }

  fclose(pfile);
}

void write_bin_ascii_data(torch::Tensor rad, RadiationBand const &band,
                          std::string fname) {
  FILE *pfile = fopen(fname.c_str(), "w");

  for (int i = 0; i < band->spec.size(0); ++i) {
    fprintf(pfile, "%13.3g%12.3g", band->spec[i][0].item().toFloat(),
            band->spec[i][1].item().toFloat());
    for (int j = 0; j < rayOutput.size(0); ++j) {
      fprintf(pfile, "%12.3f", rad[j][i].item().toFloat());
    }
  }

  fclose(pfile);
}
}  // namespace harp
