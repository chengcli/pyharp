// C/C++
#include <cstring>
#include <string>
#include <vector>

// harp
#include "vectorize.hpp"

namespace harp {
template <>
std::vector<std::string> Vectorize(const char* cstr, const char* delimiter) {
  std::vector<std::string> arr;
  char str[1028], *p;
  snprintf(str, sizeof(str), "%s", cstr);
  p = std::strtok(str, delimiter);
  while (p != NULL) {
    arr.push_back(p);
    p = std::strtok(NULL, delimiter);
  }
  return arr;
}
}  // namespace harp
