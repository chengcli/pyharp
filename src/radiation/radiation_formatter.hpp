#pragma once

// fmt
#include <fmt/format.h>

// harp
#include <harp/opacity/opacity_formatter.hpp>

#include "radiation.hpp"
#include "radiation_band.hpp"

template <>
struct fmt::formatter<harp::RadiationBandOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const harp::RadiationBandOptions& p, FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "(name = {}; solver_name = {}; opacities = {}; integration = {})",
        p.name(), p.solver_name(), p.opacities(), p.integration());
  }
};

template <>
struct fmt::formatter<harp::RadiationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const harp::RadiationOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(bands = {})", p.bands());
  }
};
