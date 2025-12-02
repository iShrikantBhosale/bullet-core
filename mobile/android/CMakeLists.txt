# CMakeLists.txt
# Android NDK Build Configuration for Bullet OS

cmake_minimum_required(VERSION 3.18.1)

project("bulletos")

# C++17 required for Bullet OS
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -march=native")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../../bq4
    ${CMAKE_SOURCE_DIR}/../..
)

# Source files
add_library(
    bulletos
    SHARED
    bullet_jni.cpp
    ../../bullet-core.cpp
)

# Link against Android log library
find_library(log-lib log)
target_link_libraries(bulletos ${log-lib})

# Strip symbols for smaller .so
set_target_properties(bulletos PROPERTIES LINK_FLAGS_RELEASE -s)
