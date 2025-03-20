include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME elements)
set(REPO_URL "https://github.com/chengcli/elements")
set(REPO_TAG "v1.1.5")
add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)

include_directories(${elements_SOURCE_DIR})
