# CMakeLists.txt

# Minimum required CMake version
cmake_minimum_required(VERSION 3.0)

# Project name
project(snnpy)

# Add source files from the 'src' folder
file(GLOB SOURCES src/*.c)

# Add source files to a library or executable
add_library(SNNpy
 src/main.c)
# or add_executable(MyExecutable ${SOURCES})

# Optionally, specify include directories
target_include_directories(SNNpy
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optionally, link external libraries
target_link_libraries(SNNpy
    PUBLIC
    SomeExternalLibrary
)
