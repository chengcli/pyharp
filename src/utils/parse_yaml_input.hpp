#pragma once

// harp
#include <yaml-cpp/yaml.h>

namespace harp {

std::string parse_unit_with_default(YAML::Node const &node);

std::pair<double, double> parse_wave_range(YAML::Node const &node);

std::string replace_pattern(std::string const &str, std::string const &pattern,
                            std::string const &replacement);

void replace_pattern_inplace(std::string &str, std::string const &pattern,
                             std::string const &replacement);

}  // namespace harp
