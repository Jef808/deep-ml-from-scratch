find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_mnist_torch test_mnist_torch.cpp)
target_link_libraries(test_mnist_torch
  PRIVATE
  core_lib
  Eigen3::Eigen
  "${TORCH_LIBRARIES}")
set_property(TARGET test_mnist_torch PROPERTY CXX_STANDARD 17)

add_executable(test_mnist_conv mnist_conv.cpp)
target_link_libraries(test_mnist_conv
  PRIVATE
  core_lib
  Eigen3::Eigen
  "${TORCH_LIBRARIES}")
set_property(TARGET test_mnist_conv PROPERTY CXX_STANDARD 17)
target_compile_definitions(test_mnist_conv PRIVATE "-DMNIST_DATA_DIR=\"${DMLFS_DATA_DIR}/MNIST/raw\"")

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_executable(dist-mnist dist_mnist.cpp)
target_link_libraries(dist-mnist
  PRIVATE
  core_lib
  Eigen3::Eigen
  ${TORCH_LIBRARIES}
  ${MPI_LIBRARIES}
  ${CMAKE_PREFIX_PATH}/lib/libc10d.a)
