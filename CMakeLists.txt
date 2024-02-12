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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Adding libraries
add_library(
  e4c 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/exceptions4c-3.0.5/src/e4c.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/exceptions4c-3.0.5/src/e4c.h
)
add_library(
  testing
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/testing-0.0.1/src/testing.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/testing-0.0.1/src/testing.h
)

# Add files that the exe will be built from
add_executable(
  SNNpy
  src/main.c
  src/neuralNetworkInit.c
  src/neuralNetworkTraining.c
  src/vectorOperations.c
  src/sigmoid.c
  src/neuralNetworkOperations.c
  src/costFunction.c
  src/neuralNetworkUtility.c
)

add_executable(
  SNNpy-tests
  tests/main.c
  tests/neuralNetworkTests.c
)

# Set the C standard to C17
target_compile_features(SNNpy PRIVATE c_std_17)

# Include the header files
target_include_directories(SNNpy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

# compiler definitions for exceptions4c compliance
add_definitions(-D E4C_THREADSAFE)

# Include the exceptions4c library
target_include_directories(SNNpy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/exceptions4c-3.0.5/src/)

target_include_directories(SNNpy-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs/testing-0.0.1/src/)

# Link the executable to the library
target_link_libraries(SNNpy e4c)

target_link_libraries(SNNpy-tests testing)


