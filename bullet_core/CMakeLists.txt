cmake_minimum_required(VERSION 3.18)
project(bullet_core LANGUAGES CXX)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_FOUND TRUE)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_FOUND FALSE)
    message(STATUS "CUDA not found. Building CPU kernels only.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture for GT 730 (CC 3.5)
set(CMAKE_CUDA_ARCHITECTURES 35)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# SIMD flags for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -msse4.2 -mfma")
endif()

# Find packages
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# CPU kernel sources
set(CPU_SOURCES
    cpp/kernels/gemm.cpp
    cpp/kernels/softmax.cpp
    cpp/kernels/rmsnorm.cpp
    cpp/kernels/layernorm.cpp
    cpp/kernels/embedding.cpp
    cpp/kernels/vector_ops.cpp
)

# CUDA kernel sources
set(CUDA_SOURCES
    cuda/gemm_cuda.cu
    cuda/softmax_cuda.cu
    cuda/rmsnorm_cuda.cu
    cuda/layernorm_cuda.cu
    cuda/vector_ops_cuda.cu
    cuda/cuda_memory.cu
    cuda/cuda_device.cu
)

# Build CPU module
pybind11_add_module(bullet_core_cpp
    cpp/bindings.cpp
    ${CPU_SOURCES}
)

target_link_libraries(bullet_core_cpp PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(bullet_core_cpp PRIVATE cpp)

# Build CUDA module
if(CUDA_FOUND)
    pybind11_add_module(bullet_core_cuda
        cuda/cuda_bindings.cpp
        ${CUDA_SOURCES}
    )

    target_link_libraries(bullet_core_cuda PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(bullet_core_cuda PRIVATE cuda ${CUDA_INCLUDE_DIRS})

    # Set CUDA properties
    set_target_properties(bullet_core_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Installation
install(TARGETS bullet_core_cpp DESTINATION .)
if(CUDA_FOUND)
    install(TARGETS bullet_core_cuda DESTINATION .)
endif()
