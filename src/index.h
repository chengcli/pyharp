#pragma once

namespace harp {

namespace index {
// atm (legacy)
constexpr int ITM = 0;
constexpr int IPR = 1;
constexpr int ICX = 2;

// optical variables
constexpr int IEX = 0;  //! extinction cross section
constexpr int ISS = 1;  //! single scattering albedo
constexpr int IPM = 2;  //! phase moments

// flux variables
constexpr int IUP = 0;  //! upward
constexpr int IDN = 1;  //! downward
};  // namespace index

}  // namespace harp
