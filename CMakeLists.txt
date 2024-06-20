set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

cmake_minimum_required(VERSION 3.28)
project(dml-from-scratch VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenMP REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include(eigen)
include(spdlog)
include(Catch2)

set(DMLFS_DATA_DIR ${CMAKE_SOURCE_DIR}/data)
set(DMLFS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(DMLFS_TESTS_DIR ${DMLFS_SOURCE_DIR}/tests)

##############################################################################
# Set the properties each single targets should inherit within this project. #
##############################################################################
add_library(globals INTERFACE ${CMAKE_SOURCE_DIR}/src/CommonMacros.h)
target_compile_definitions(globals INTERFACE "-DDATA_DIR=\"${DMLFS_DATA_DIR}\"")
target_include_directories(globals INTERFACE ${DMLFS_SOURCE_DIR})
link_libraries(globals)

####################################################
# The core library implementing a network's layers #
####################################################
add_library(core_lib
  ${CMAKE_SOURCE_DIR}/src/network/layer.h
  ${CMAKE_SOURCE_DIR}/src/network/layer.cpp
  ${CMAKE_SOURCE_DIR}/src/network/network.h
  ${CMAKE_SOURCE_DIR}/src/network/network.cpp
  ${CMAKE_SOURCE_DIR}/src/network/activation.h
  ${CMAKE_SOURCE_DIR}/src/network/activation.cpp
  ${CMAKE_SOURCE_DIR}/src/network/initializer.h
  ${CMAKE_SOURCE_DIR}/src/network/initializer.cpp
  ${CMAKE_SOURCE_DIR}/src/network/optimizer.h
  ${CMAKE_SOURCE_DIR}/src/network/optimizer.cpp
)
target_link_libraries(core_lib PRIVATE Eigen3::Eigen)

#########################################
# Setting executables for my unit tests #
#########################################
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
enable_testing()

add_executable(test_setup ${CMAKE_SOURCE_DIR}/src/tests/test_setup.cc)
target_link_libraries(test_setup PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)
catch_discover_tests(test_setup)

add_executable(test_layer ${CMAKE_SOURCE_DIR}/src/tests/test_layer.cc)
target_link_libraries(test_layer PRIVATE core_lib Catch2::Catch2WithMain Eigen3::Eigen)
catch_discover_tests(test_layer)

add_executable(test_iris ${CMAKE_SOURCE_DIR}/src/tests/test_iris.cc src/datautils/csv.h)
target_link_libraries(test_iris PRIVATE core_lib Catch2::Catch2WithMain Eigen3::Eigen)
catch_discover_tests(test_iris)

add_executable(test_data_preprocessing ${DMLFS_TESTS_DIR}/test_data_preprocessing.cc)
target_link_libraries(test_data_preprocessing PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)

# add_executable(test_mnist ${CMAKE_SOURCE_DIR}/src/tests/test_mnist.cc src/datautils/csv.h)
# target_link_libraries(test_mnist PRIVATE core_lib Catch2::Catch2WithMain Eigen3::Eigen)
# catch_discover_tests(test_mnist)

#################
# Add examples  #
#################
#add_subdirectory(${DMLFS_SOURCE_DIR}/examples)

add_subdirectory(${DMLFS_SOURCE_DIR}/tests)
