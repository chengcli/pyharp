# Install rtsolver submodule
#
#   library: librtsolver_debug.o
#   library: librtsolver_release.o
#
# Define the following symbols
#
#   rtsolver_debug
#   rtsolver_release
#
# Compile the following submodules
#
#   cdisort : Discrete Ordinate Method solver, cdisort-2.1.3

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel rtsolver)
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

target_link_libraries(${namel}_${buildl}
		PRIVATE
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    )

add_subdirectory(cdisort213)
