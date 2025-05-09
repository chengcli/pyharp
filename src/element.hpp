#pragma once

// C/C++
#include <map>
#include <string>
#include <vector>

namespace harp {

//! \return a vector of all element symbols
const std::vector<std::string>& element_symbols();

//! \return a vector of all element names
const std::vector<std::string>& element_names();

//! \return a vector of all element atomic numbers
const std::map<std::string, double>& element_weights();

//! \brief Get the atomic weight of an element by name
/*!
 * \param ename The name of the element
 * \return The atomic weight of the element
 */
double get_element_weight(const std::string& ename);

//! \brief Get the atomic weight of an element by atomic number
/*!
 * \param atomicNumber The atomic number of the element
 * \return The atomic weight of the element
 */
double get_element_weight(int atomicNumber);

//! \brief Get the element symbol from the element name
/*!
 * \param ename The name of the element
 * \return The element symbol
 */
std::string get_element_symbol(const std::string& ename);

//! \brief Get the element symbol from the atomic number
/*!
 * \param atomicNumber The atomic number of the element
 * \return The element symbol
 */
std::string get_element_symbol(int atomicNumber);

//! \brief Get the element name from the element symbol
/*!
 * \param esym The symbol of the element
 * \return The element name
 */
std::string get_element_name(const std::string& ename);

//! \brief Get the element name from the atomic number
/*!
 * \param atomicNumber The atomic number of the element
 * \return The element name
 */
std::string get_element_name(int atomicNumber);

//! \brief Get the atomic number of an element by name
/*!
 * \param ename The name of the element
 * \return The atomic number of the element
 */
int get_atomic_number(const std::string& ename);

//! \return total number of elements defined
size_t num_elements_defined();

//! \return total number of isotopes defined
size_t num_isotopes_defined();

}  // namespace harp
