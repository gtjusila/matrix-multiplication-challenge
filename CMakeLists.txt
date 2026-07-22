cmake_minimum_required(VERSION 3.28)

project(advanced-matrix-multiplication LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CCCL REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES
    native
    CACHE STRING "")
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

add_library(matrix_multiply STATIC src/kernels/basic_multiply.cu
                                   src/kernels/cublas_multiply.cu)
target_link_libraries(matrix_multiply PRIVATE CUDA::cublas)

add_executable(runner src/runner.cpp src/check_implementation.cu)
target_link_libraries(runner PRIVATE CUDA::cudart CUDA::curand CCCL::CCCL
                                     matrix_multiply)
