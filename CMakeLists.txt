# CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# Project name
project(
  SNNpy
  VERSION 2.0.0
  LANGUAGES C
)  

# Doxygen docs
find_package(Doxygen)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
endif()

# Adding libraries

add_library(
  testing STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/testing-0.0.1/src/testing.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/testing-0.0.1/src/testing.h
)

add_library(
  SNNpy SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkInit.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkTraining.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vectorOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sigmoid.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/costFunction.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkUtility.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/
)

target_include_directories(SNNpy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/)

# Add executable

add_executable(
  SNNpy-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkInit.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkTraining.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vectorOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sigmoid.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/costFunction.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkUtility.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/
)
target_include_directories(SNNpy-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/)
target_include_directories(SNNpy-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/testing-0.0.1/src)


set_target_properties(SNNpy-tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

set_target_properties(SNNpy PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/libs
)

# Set the C standard to C17
target_compile_features(SNNpy PRIVATE c_std_17)

# Link the executable to the library
target_link_libraries(SNNpy-tests testing)


