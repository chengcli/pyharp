#pragma once

// fmt
#include <fmt/format.h>

// harp
#include <harp/opacity/opacity_formatter.hpp>

#include "radiation.hpp"
#include "radiation_band.hpp"

//! \brief Formatter specialization for RadiationBandOptions
/*!
 * This formatter enables `fmt::format` to print `RadiationBandOptions`
 * objects. The output format is:
 * `(name = <name>; solver_name = <solver>; opacities = <opacity_list>)`
 */
template <>
struct fmt::formatter<harp::RadiationBandOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const harp::RadiationBandOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(),
                          "(name = {}; solver_name = {}; opacities = {})",
                          p->name(), p->solver_name(), p->opacities());
  }
};

//! \brief Formatter specialization for RadiationOptions
/*!
 * This formatter enables `fmt::format` to print `RadiationOptions`
 * objects. The output format is: `(bands = <bands_list>)`
 */
template <>
struct fmt::formatter<harp::RadiationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const harp::RadiationOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(bands = {})", p->bands());
  }
};
