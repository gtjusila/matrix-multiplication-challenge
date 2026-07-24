#pragma once

namespace tiled_multiply {
template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n);

}  // namespace tiled_multiply
