#pragma once

// fmt
#include <fmt/format.h>

template <typename T>
struct fmt::formatter<std::map<std::string, T>> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::map<std::string, T>& p, FormatContext& ctx) const {
    std::string result = "{\n";
    for (auto const& [key, value] : p) {
      result += fmt::format("\t{}: {},", key, value);
      result += "\n";
    }
    result += "}";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};

template <typename T>
struct fmt::formatter<std::vector<T>> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::vector<T>& p, FormatContext& ctx) const {
    std::string result = "(";
    for (size_t i = 0; i < p.size(); ++i) {
      result += fmt::format("{}", p[i]);
      if (i < p.size() - 1) {
        result += ", ";
      }
    }
    result += ")";
    return fmt::format_to(ctx.out(), "{}", result);
  }
};
