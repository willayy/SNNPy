# Minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# Project name
project(
  SNN
  LANGUAGES C
  VERSION 4.0.0
  DESCRIPTION "A neural network model implemented in C"
)  

# Set the C standard to C23
set(CMAKE_C_STANDARD 23)

# Set the C standard required by the project
set(CMAKE_C_STANDARD_REQUIRED ON)

# When compile is set to debug with -g flag compile the tests into an executable
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(BUILD_TESTS ON)
else()
  set(BUILD_TESTS OFF)
endif()

# wmath soure files
set(
  WMATH_SOURCE 
  extclibs/wmath/src/activationFunctions.c
  extclibs/wmath/src/costFunctions.c
  extclibs/wmath/src/randomValueGenerator.c
)

# wmath header directory
include_directories(WMATH_INCLUDE extclibs/wmath/src/include)

# Add the wmath library
add_library(wmath ${WMATH_SOURCE})

# Include the source directory
target_include_directories(wmath PRIVATE ${WMATH_INCLUDE})

# if debug build tests
if (BUILD_TESTS)
  
  # wtest source files
  set(
    WTEST_SOURCE
    extclibs/wtest/src/testing.c
  )

  # wtest header directory
  include_directories(WTEST_INCLUDE extclibs/wtest/src/include)

  # Add the wtest library
  add_library(wtest ${WTEST_SOURCE})

  # Include the header directory
  target_include_directories(wtest PRIVATE ${WTEST_INCLUDE})

  # test source files
  set(
    TEST_SOURCE
    test/test.c
  )

  # header directory for the tests
  include_directories(TEST_INCLUDE src/include)

  # Add the test executable
  add_executable(test ${TEST_SOURCE})

  # include the heaeder directory
  target_include_directories(test PRIVATE ${TEST_SOURCE})

  # Link the wtest library to the test executable
  target_link_libraries(test wtest)
  target_link_libraries((test wmath))

endif()