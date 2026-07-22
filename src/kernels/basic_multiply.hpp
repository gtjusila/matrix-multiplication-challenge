#pragma once

#include <cstdlib>
namespace basic_multiply {
template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, std::size_t m, std::size_t k,
                     std::size_t n);

}  // namespace basic_multiply
