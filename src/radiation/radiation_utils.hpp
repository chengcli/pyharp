#pragma once

// torch
#include <torch/torch.h>

// harp
#include "radiation.hpp"

namespace harp {
//! \brief Get the number of grids in the outgoing ray directions
//! \return [phi grids, mu grids]
std::pair<std::vector<float>, std::vector<float>> get_direction_grids(
    std::vector<std::pair<float, float>> const& dirs);

//! \brief Parse radiation direction string
//!
//! First number is the polar angle (degrees), second number is the azimuthal
//! angle (degrees)
//!
//! \param[in] str radiation direction string, e.g. (45, 30)
//! \return radiation direction in (mu, phi)
torch::Tensor parse_radiation_direction(std::string const& str);

//! \brief Parse radiation directions string, sperated by comma
//!
//! Example input string: "(45, 30), (45, 60)"
//! \param[in] str radiation directions string
torch::Tensor parse_radiation_directions(std::string const& str);

RadiationOptions parse_radiation_flags(std::string const& str);

//! \brief Get the phase function moment
//!
//! \param[in] phase_func phase function name
//! \param[in] npmom number of phase function moments
//! \param[in] gg1 Henyey-Greenstein asymmetry factor
torch::Tensor get_phase_moment(std::string const& phase_func, int npmom,
                               float gg1, float gg2 = 0., float ff = 0.);

//! \brief Parse the unit of the spectral grid
//!
//! \return [wavenumber, wavelength, frequency]
// std::string parse_unit_with_default(YAML::Node const& my);

//! \brief Parse the spectral range from a YAML node
//!
//! \return [wave min, wave max]
// std::pair<float, float> parse_wave_range(YAML::Node const& my);

}  // namespace harp
