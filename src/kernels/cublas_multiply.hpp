#pragma once

#include <concepts>
namespace cublas_multiply {
template <typename T>
  requires std::same_as<T, float> || std::same_as<T, double>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n);

}  // namespace cublas_multiply
