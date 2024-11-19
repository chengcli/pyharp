# A small macro used for setting up the build of a test.
#
# Usage: setup_test(name)

macro(setup_test namel)
  string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
  string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(${namel}.${buildl}
                        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${HARP_INCLUDE_DIR}
            ${NETCDF_INCLUDES}
            SYSTEM
            ${TORCH_INCLUDE_DIR}
            SYSTEM
            ${TORCH_API_INCLUDE_DIR})

  target_link_libraries(${namel}.${buildl} gtest_main harp::harp)

  add_test(NAME ${namel}.${buildl} COMMAND ${namel}.${buildl})
endmacro()