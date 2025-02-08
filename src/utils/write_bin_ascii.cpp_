// C/C++
#include <fstream>
#include <iostream>

// harp
#include "parse_radiation_direction.hpp"  // rad2deg
#include "write_bin_ascii.hpp"

namespace harp {
void write_bin_ascii_header(RadiationBand const &band, std::string fname) {
  FILE *pfile = fopen(fname.c_str(), "w");

  fprintf(pfile, "# Bin Radiances of Band %s: %.3g - %.3g\n",
          band->name().c_str(), band->options.wave_lower().front(),
          band->options.wave_upper().back());
  ;
  auto const &rayOutput = band->rayOutput;
  fprintf(pfile, "# Ray output size: %lld\n", rayOutput.size(0));

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
    fprintf(pfile, "%12s%d", "Radiance", i + 1);
  }

  fclose(pfile);
}

void write_bin_ascii_data(torch::Tensor rad, RadiationBand const &band,
                          std::string fname) {
  FILE *pfile = fopen(fname.c_str(), "w");

  for (int i = 0; i < band->weight.size(0); ++i) {
    fprintf(pfile, "%13.3g%13.3g%12.3g", band->options.wave_lower()[i],
            band->options.wave_upper()[i], band->weight[i].item().toFloat());
    for (int j = 0; j < band->rayOutput.size(0); ++j) {
      fprintf(pfile, "%12.3f", rad[j][i].item().toFloat());
    }
  }

  fclose(pfile);
}
}  // namespace harp
