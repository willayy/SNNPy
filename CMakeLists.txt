# CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.20)


# Project name
project(
  SNN
  LANGUAGES C
  VERSION 3.0.0
  DESCRIPTION "A neural network model implemented in C"
)  

# Adding the testing library
add_library(
  testing STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/testing.c
  ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/testing.h
)

# Set the output directory for the archive testing library
set_target_properties(testing PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Add shared library
add_library(
  ${PROJECT_NAME} SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nnMemManagement.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkInit.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkTraining.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/activationFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkOperations.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/costFunctions.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/neuralNetworkUtility.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/randomValueGenerator.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tests.c
)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extclibs/testing-0.0.1/src/)

if (WIN32) # Windows output directory for shared library
  set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/sharedlib)
  set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/sharedlib)
endif()

if (UNIX) # Linux or MacOS output directory for shared library
  set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/snnpy/sharedlib)
endif()

# Set the C standard to C17
 target_compile_features(${PROJECT_NAME} PRIVATE c_std_17)

# Link the executable to the library
target_link_libraries(${PROJECT_NAME} testing)



