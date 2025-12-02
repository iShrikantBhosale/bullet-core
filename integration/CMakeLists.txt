cmake_minimum_required(VERSION 3.10)
project(BulletIntegration)

set(CMAKE_CXX_STANDARD 17)

# CLI Tool
add_executable(bullet_cli bullet_cli.cpp)

# Python Bindings (Optional, requires pybind11)
find_package(pybind11 CONFIG)
if(pybind11_FOUND)
    pybind11_add_module(bullet_bindings bullet_python_bindings.cpp)
else()
    message(STATUS "pybind11 not found, skipping python bindings build via CMake. Use setup.py instead.")
endif()
