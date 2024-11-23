#include "radiation_utils.hpp"

#include "vectorize.hpp"

namespace harp {
std::string parse_unit_with_default(YAML::Node const &my) {
  std::string units = my["units"] ? my["units"].as<std::string>() : "cm-1";

  if (units == "cm-1") {
    return "wavenumber";
  } else if (units == "um" || units == "nm" || units == "A") {
    return "wavelength";
  } else if (units == "GHz") {
    return "frequency";
  } else {
    throw std::runtime_error(
        "parse_unit_with_default::unknown spectral unit type");
  }
}

std::pair<float, float> parse_wave_range(YAML::Node const &my) {
  auto unit = parse_unit_with_default(my);

  char str[80];
  snprintf(str, sizeof(str), "%s-%s", unit.c_str(), "range");

  if (!my[str]) {
    throw std::runtime_error("parse_wave_range::" + str + " not found");
  }

  /// wavenumber-range, wavelength-range, frequency-range, etc
  float wmin = my[str][0].as<float>();
  float wmax = my[str][1].as<float>();
  if (wmin > wmax) {
    throw std::runtime_error("parse_wave_range::wmin > wmax");
  }

  return std::make_pair(wmin, wmax);
}
}  // namespace harp
