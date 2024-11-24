# Install harp library
#
#   library: libharp_debug.o
#   library: libharp_release.o
#
# Define the following symbols
#
#   HARP_INCLUDE_DIR
#   CDISORT_INCLUDE_DIR
#   harp::harp
#   harp::cdisort
#
# Normal usage would be:
#
#   include_directories(${HARP_INCLUDE_DIR})
#   target_link_libraries(harp::harp)
#
# Compile the following submodules
#
#   utils     : utility functions
#   opacity   : opacity providers
#   radiation : radiation and radiation bands
#   rtsolver  : radiative transfer solvers

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}
  ${NETCDF_INCLUDES}
  ${PNETCDF_INCLUDE_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel harp)
string(TOUPPER ${namel} nameu)

file(GLOB src_files *.cpp)

add_library(${namel}_${buildl}
    OBJECT
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

add_subdirectory(utils)
add_subdirectory(opacity)
add_subdirectory(radiation)
add_subdirectory(rtsolver)

add_library(${namel}_${buildl}_final
  STATIC
  $<TARGET_OBJECTS:${namel}_${buildl}>
  $<TARGET_OBJECTS:utils_${buildl}>
  $<TARGET_OBJECTS:opacity_${buildl}>
  $<TARGET_OBJECTS:radiation_${buildl}>
  $<TARGET_OBJECTS:rtsolver_${buildl}>
  $<TARGET_OBJECTS:cdisort_${buildl}>
  )

add_library(harp::harp ALIAS ${namel}_${buildl}_final)

set(HARP_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  CACHE INTERNAL "harp include directory")
