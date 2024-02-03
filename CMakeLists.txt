# CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.0)

# Project name
project(SNNpy
VERSION 1.0
LANGUAGES C)

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
Main src/main.c
src/neuralNetworkInit.c
src/neuralNetworkTraining.c
src/vectorOperations.c
src/sigmoid.c
src/neuralNetworkOperations.c
src/costFunction.c
src/neuronUtility.c
)

target_compile_features(Main PRIVATE c_std_17)

target_include_directories(Main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
