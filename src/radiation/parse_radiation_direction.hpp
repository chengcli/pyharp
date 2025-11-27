#pragma once

#include <cmath>
#include <string>

namespace at {
class Tensor;
}

namespace torch {
using Tensor = at::Tensor;
}

namespace harp {

//! \brief Convert radians to degrees
/*!
 * \tparam T The numeric type (typically float or double)
 * \param phi The angle in radians
 * \return The angle converted to degrees
 */
template <typename T>
T rad2deg(T phi) {
  return phi * 180. / M_PI;
}

//! \brief Convert degrees to radians
/*!
 * \tparam T The numeric type (typically float or double)
 * \param phi The angle in degrees
 * \return The angle converted to radians
 */
template <typename T>
T deg2rad(T phi) {
  return phi * M_PI / 180.;
}

//! \brief Parse a single radiation direction string
/*!
 * This function parses a direction string in the format "(theta,phi)"
 * where theta is the polar angle and phi is the azimuthal angle,
 * both in degrees.
 *
 * The function converts the angles as follows:
 * - theta is converted to mu = cos(deg2rad(theta))
 * - phi is converted from degrees to radians
 *
 * \param str A direction string in the format "(theta,phi)"
 * \return A 1D tensor of shape (2,) containing [mu, phi_radians]
 */
torch::Tensor parse_radiation_direction(std::string const &str);

//! \brief Parse multiple radiation direction strings
/*!
 * This function parses a string containing multiple radiation directions
 * separated by whitespace or commas. Each direction should be in the
 * format "(theta,phi)" where theta is the polar angle and phi is the
 * azimuthal angle, both in degrees.
 *
 * Each direction is converted using parse_radiation_direction.
 *
 * \param str A string containing multiple direction specifications
 * \return A 2D tensor of shape (n, 2) where each row contains [mu, phi_radians]
 */
torch::Tensor parse_radiation_directions(std::string const &str);
}  // namespace harp
