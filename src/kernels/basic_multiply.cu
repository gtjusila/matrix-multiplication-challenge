#include "basic_multiply.hpp"
#include "helper.cuh"

namespace basic_multiply {

template <typename T>
__global__ void matrix_multiply_kernel(const T* __restrict__ A,
                                       const T* __restrict__ B,
                                       T* __restrict__ C, int m, int k, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int tx = idx % n;
  int ty = idx / n;
  if (tx < n && ty < m) {
    T res = 0;
    for (int i = 0; i < k; ++i) {
      res += A[ID2X(ty, i, k)] * B[ID2X(i, tx, n)];
    }
    C[ID2X(ty, tx, n)] = res;
  }
};

template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n) {
  auto thread_count = 256;
  auto block_count = (m * n + 256 - 1) / 256;
  matrix_multiply_kernel<<<block_count, thread_count>>>(A, B, C, m, k, n);
}

template void matrix_multiply<float>(const float* A, const float* B, float* C,
                                     int m, int k, int n);
template void matrix_multiply<double>(const double* A, const double* B,
                                      double* C, int m, int k, int n);
}  // namespace basic_multiply
