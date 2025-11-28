#pragma once

// C/C++
#include <sstream>

// fmt
#include <fmt/format.h>

// harp
#include "integrator.hpp"

template <>
struct fmt::formatter<harp::IntegratorWeight> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for IntegratorOptions
  template <typename FormatContext>
  auto format(const harp::IntegratorWeight& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "({}, {}, {})", p.wght0(), p.wght1(),
                          p.wght2());
  }
};

template <>
struct fmt::formatter<harp::IntegratorOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for IntegratorOptions
  template <typename FormatContext>
  auto format(const harp::IntegratorOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p->report(ss);
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};
