#pragma once

#include <format>
#include <iostream>

#include "cuda_runtime.h"

#define CUDA_CHECK_(x)                                                         \
  do {                                                                         \
    if (auto err_ = (x); err_ != cudaSuccess) {                                \
      std::cout << std::format("Error at kernel call {}:{} {}:{}\n", __FILE__, \
                               __LINE__, cudaGetErrorName(err_),               \
                               cudaGetErrorString(err_));                      \
    }                                                                          \
  } while (false)
#define KERNEL_CALL(...)                  \
  do {                                    \
    __VA_ARGS__;                          \
    CUDA_CHECK_(cudaGetLastError());      \
    CUDA_CHECK_(cudaDeviceSynchronize()); \
  } while (false)
