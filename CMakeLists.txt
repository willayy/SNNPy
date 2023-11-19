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

# Add source files to a library that can be linked with other code
add_library(SNNpy src/neuron.c)

# Add files that the exe will be built from
add_executable(Main src/main.c)

