# CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# Project name
project(
  SNNPy
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

# Adding the testing library
add_library(
  testing STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/testing.c
    ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/testing.h
)

# Add the executable for testsuite
add_executable(
  SNNPy-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMemManagement.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkInit.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkTraining.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/activationFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/costFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkUtility.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/randomValueGenerator.c
)
target_include_directories(SNNPy-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/)
target_include_directories(SNNPy-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/)

# Add shared library
add_library(
  SNNPy SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMemManagement.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkInit.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkTraining.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/activationFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/costFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkUtility.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/randomValueGenerator.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.c
)
target_include_directories(SNNPy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/)
target_include_directories(SNNPy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/)

add_executable(
  SNNPy-benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMemManagement.c
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmarkTest.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkInit.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkTraining.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/activationFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/costFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkUtility.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/randomValueGenerator.c
)
target_include_directories(SNNPy-benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/)

# Set the output directory for the testsuite
set_target_properties(SNNPy-tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# Set the output directory for the benchmark
set_target_properties(SNNPy-benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

# Set the output directory for the archive testing library
set_target_properties(testing PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

if (WIN32) # Windows output directory for shared library
  set_target_properties(SNNPy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/sharedlib)
  set_target_properties(SNNPy PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/sharedlib)
endif()

if (UNIX) # Linux or MacOS output directory for shared library
  set_target_properties(SNNPy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/sharedlib)
endif()

# Set the C standard to C17
target_compile_features(SNNPy PRIVATE c_std_17)

# Link the executable to the library
target_link_libraries(SNNPy-tests testing)


