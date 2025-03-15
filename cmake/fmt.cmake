include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME fmt)
set(REPO_URL "https://github.com/fmtlib/fmt")
set(REPO_TAG "11.1.2")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${fmt_SOURCE_DIR}/include)
