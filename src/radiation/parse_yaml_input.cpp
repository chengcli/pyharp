// C/C++
#include <stdexcept>

// harp
#include "parse_yaml_input.hpp"

namespace harp {

std::string parse_unit_with_default(YAML::Node const &node) {
  std::string units = node["units"] ? node["units"].as<std::string>() : "cm-1";

  if (units == "cm-1") {
    return "wavenumber";
  } else if (units == "um" || units == "nm" || units == "A") {
    return "wavelength";
  } else if (units == "GHz") {
    return "frequency";
  } else {
    throw std::runtime_error("unknown spectral unit type");
  }
}

std::pair<double, double> parse_wave_range(YAML::Node const &node) {
  auto unit = parse_unit_with_default(node);

  char str[80];
  snprintf(str, sizeof(str), "%s-%s", unit.c_str(), "range");

  if (!node[str]) {
    throw std::runtime_error("missing spectral range");
  }

  /// wavenumber-range, wavelength-range, frequency-range, etc
  double wmin = node[str][0].as<double>();
  double wmax = node[str][1].as<double>();

  if (wmin > wmax) {
    throw std::runtime_error("invalid spectral range");
  }

  return std::make_pair(wmin, wmax);
}

std::string replace_pattern(std::string const &str, std::string const &pattern,
                            std::string const &replacement) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(pattern, pos)) != std::string::npos) {
    result.replace(pos, pattern.length(), replacement);
    pos += replacement.length();
  }
  return result;
}

void replace_pattern_inplace(std::string &str, std::string const &pattern,
                             std::string const &replacement) {
  size_t pos = 0;
  while ((pos = str.find(pattern, pos)) != std::string::npos) {
    str.replace(pos, pattern.length(), replacement);
    pos += replacement.length();
  }
}

}  // namespace harp
