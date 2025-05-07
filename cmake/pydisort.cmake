include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME disort)
set(REPO_URL "https://github.com/zoeyzyhu/pydisort")
set(REPO_TAG "v1.1.2")
add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)

include_directories(${disort_SOURCE_DIR})
