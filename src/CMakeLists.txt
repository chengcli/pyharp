# Install harp library
#
#   library: libharp_debug.o
#   library: libharp_release.o
#
# Define the following symbols
#
#   HARP_INCLUDE_DIR
#   pyharp::harp
#
# Normal usage would be:
#
#   include_directories(${HARP_INCLUDE_DIR})
#   target_link_libraries(pyharp::harp)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${CMAKE_BINARY_DIR}
  ${NETCDF_INCLUDES}
  SYSTEM ${DISORT_INCLUDE_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel harp)
string(TOUPPER ${namel} nameu)

file(GLOB src_files
    *.cpp
    math/*.cpp
    utils/*.cpp
    opacity/*.cpp
    radiation/*.cpp
    #rtsolver/*.cpp
    integrator/*.cpp
    )

add_library(${namel}_${buildl}
    SHARED
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    fmt::fmt
    yaml-cpp::yaml-cpp
    ${NETCDF_LIBRARIES}
    ${DISORT_LIBRARY}
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    )

add_library(pyharp::harp ALIAS ${namel}_${buildl})

set(HARP_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE INTERNAL "harp include directory")
