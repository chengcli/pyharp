include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

# yaml-cpp enables a few helper binaries and contrib sources by default.
# They are not used by pyharp and add noticeable CI build time.
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

set(patch_command
    git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/01.yamlpp.patch)

set(PACKAGE_NAME yaml-cpp)
set(REPO_URL "https://github.com/jbeder/yaml-cpp")
set(REPO_TAG "0.8.0")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "${patch_command}" ON)
include_directories(${yaml-cpp_SOURCE_DIR}/include)
