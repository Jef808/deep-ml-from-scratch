Include(FetchContent)

FetchContent_Declare(
  OpenCV
  GIT_REPOSITORY    https://github.com/opencv/opencv
  GIT_TAG           4.9.0
)

FetchContent_MakeAvailable(OpenCV)
