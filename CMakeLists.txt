# CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.0)

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

# Set the C standard to C17
target_compile_features(SNNpy PRIVATE c_std_17)

# Include the header files
target_include_directories(SNNpy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# compiler definitions for exceptions4c compliance
add_definitions(-D E4C_THREADSAFE)

# Adding libraries
add_library(
  e4c 
    ${CMAKE_CURRENT_SOURCE_DIR}/externalCLibs/exceptions4c-3.0.5/src/e4c.c
    ${CMAKE_CURRENT_SOURCE_DIR}/externalCLibs/exceptions4c-3.0.5/src/e4c.h
)
add_library(
  simpleCtests
    ${CMAKE_CURRENT_SOURCE_DIR}/externalCLibs/simpleCtests-0.0.1/src/simpleCtests.c
    ${CMAKE_CURRENT_SOURCE_DIR}/externalCLibs/simpleCtests-0.0.1/src/simpleCtests.h
)

# Include the exceptions4c library
target_include_directories(SNNpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/externalCLibs/exceptions4c-3.0.5/src/)
target_include_directories(SNNpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/externalCLibs/simpleCtests-0.0.1/src/)

# Link the executable to the library
target_link_libraries(SNNpy e4c simpleCtests)


