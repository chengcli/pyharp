#pragma once

// C/C++
#include <map>
#include <sstream>
#include <string>

// torch
#include <torch/torch.h>

namespace harp {

//! Map from string names to doubles. Used for defining species mole/mass
//! fractions, elemental compositions, and reaction stoichiometries.
using Composition = std::map<std::string, double>;

//! \brief Get the composition map of a compound from its formula
/*!
 * \param formula The formula of the compound
 * \return The composition map of the compound
 */
Composition get_composition(const std::string& formula);

//! \brief Get the molecular weight of a compound from its compound map
/*!
 * \param composition The composition map of the compound
 * \return The molecular weight of the compound
 */
double get_compound_weight(const Composition& composition);

//! \brief Normalize a composition map so its values sum to unity
/*!
 * \param composition Unnormalized composition map
 * \return A normalized composition map whose values sum to 1
 */
inline Composition normalize_composition(const Composition& composition) {
  double total = 0.0;
  for (auto const& [name, value] : composition) total += value;

  TORCH_CHECK(total > 0.0, "Composition sum must be positive");

  Composition normalized;
  for (auto const& [name, value] : composition) {
    normalized[name] = value / total;
  }
  return normalized;
}

//! \brief Format a composition map as a comma-separated string
/*!
 * Example output: ``H2:0.9,He:0.1``
 *
 * \param composition Composition map
 * \return Comma-separated composition string
 */
inline std::string format_composition(const Composition& composition) {
  std::ostringstream oss;
  bool first = true;
  for (auto const& [name, value] : composition) {
    if (!first) oss << ",";
    first = false;
    oss << name << ":" << value;
  }
  return oss.str();
}

}  // namespace harp
