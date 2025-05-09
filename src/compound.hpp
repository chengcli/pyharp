#pragma once

// C/C++
#include <map>
#include <string>

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

}  // namespace harp
