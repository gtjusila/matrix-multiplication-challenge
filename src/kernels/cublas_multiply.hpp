#pragma once

#include <concepts>
#include <cstdlib>
namespace cublas_multiply {
template <typename T>
  requires std::same_as<T, float> || std::same_as<T, double>
void matrix_multiply(const T* A, const T* B, T* C, std::size_t m, std::size_t k,
                     std::size_t n);

}  // namespace cublas_multiply
