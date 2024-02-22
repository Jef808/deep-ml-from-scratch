include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v2.x
  )
# FetchContent_GetProperties(spdlog)
# if (NOT spdlog_POPULATED)
#   FetchContent_Populate(spdlog)
#   add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
# endif()
FetchContent_MakeAvailable(spdlog)
#Fetch_Content_GetProperties(spdlog)
#if (NOT spdlog_POPULATED)
#  FetchContent_Populate(spdlog)
#  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
#endif()
