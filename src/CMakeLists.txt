# Install harp library
#
#   library: libharp_debug.o
#   library: libharp_release.o
#
# Define the following symbols
#
#   HARP_INCLUDE_DIR
#   CDISORT_INCLUDE_DIR
#   HARP_LIBRARY_DEBUG
#   HARP_LIBRARY_RELEASE
#   harp::cdisort
#
# Normal usage would be:
#
#   include_directories(${HARP_INCLUDE_DIR})
#   target_link_libraries(${HARP_LIBRARY_DEBUG})
#   target_link_libraries(${HARP_LIBRARY_RELEASE})
#
# Compile the following submodules
#
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

file(GLOB src_files
    *.cpp
    math/*.cpp
    utils/*.cpp
    opacity/*.cpp
    radiation/*.cpp
    )

add_library(${namel}_${buildl}
    STATIC
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
		PRIVATE
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    )

add_subdirectory(rtsolver)

set(HARP_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  CACHE INTERNAL "harp include directory")

set(HARP_LIBRARY_${buildu}
  harp_${buildl}
  rtsolver_${buildl}
  harp::cdisort
  CACHE INTERNAL "harp library")
