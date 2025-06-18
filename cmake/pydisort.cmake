include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME pydisort)
set(REPO_URL "https://github.com/zoeyzyhu/pydisort")
set(REPO_TAG "v1.2.3")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${pydisort_SOURCE_DIR})
