// C/C++
#include <algorithm>
#include <cctype>
#include <string>

// harp
#include "strings.hpp"

namespace harp {

std::string to_lower_copy(const std::string& str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower_str;
}

std::string trim_copy(const std::string& str) {
  // Find first non-whitespace character
  auto start = std::find_if(str.begin(), str.end(),
                            [](unsigned char ch) { return !std::isspace(ch); });

  // Find last non-whitespace character
  auto end = std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
               return !std::isspace(ch);
             }).base();

  // Ensure valid range before constructing a new string
  return (start < end) ? std::string(start, end) : std::string();
}

}  // namespace harp
