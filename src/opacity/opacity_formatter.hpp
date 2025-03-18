#pragma once

// fmt
#include <fmt/format.h>

// harp
#include <harp/harp_formatter.hpp>

#include "attenuator_options.hpp"

template <>
struct fmt::formatter<harp::AttenuatorOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const harp::AttenuatorOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(),
                          "(type = {}; opacity_files = {}; species_ids = {})",
                          p.type(), p.opacity_files(), p.species_ids());
  }
};
