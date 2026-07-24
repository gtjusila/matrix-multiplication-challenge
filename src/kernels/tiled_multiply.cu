
#include "helper.cuh"
#include "tiled_multiply.hpp"

namespace tiled_multiply {

static constexpr int kTileSize = 32;
template <typename T, int TILE_SIZE>
__global__ void matrix_multiply_kernel(const T* __restrict__ A,
                                       const T* __restrict__ B,
                                       T* __restrict__ C, int m, int k, int n) {
  __shared__ T a_s[TILE_SIZE][TILE_SIZE];
  __shared__ T b_s[TILE_SIZE][TILE_SIZE];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bCol = blockIdx.x * TILE_SIZE;
  int bRow = blockIdx.y * TILE_SIZE;

  auto tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;
  T acc = 0;
  for (int i = 0; i < tile_count; ++i) {
    // Computing tile ( y, x) of C  for each i collectively get tile (y, i) from
    // A and (i, x) from b;
    if (bRow + ty < m && i * TILE_SIZE + tx < k) {
      a_s[ty][tx] = A[ID2X(bRow + ty, i * TILE_SIZE + tx, k)];
    } else {
      a_s[ty][tx] = 0;
    }
    if (TILE_SIZE * i + ty < k && bCol + tx < n) {
      b_s[ty][tx] = B[ID2X(TILE_SIZE * i + ty, bCol + tx, n)];
    } else {
      b_s[ty][tx] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < TILE_SIZE; ++j) {
      acc += a_s[ty][j] * b_s[j][tx];
    }
    __syncthreads();
  }

  if (bRow + ty < m && bCol + tx < n) {
    C[ID2X(bRow + ty, bCol + tx, n)] = acc;
  }
};

template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n) {
  dim3 thread_layout(kTileSize, kTileSize);
  dim3 block_layout((n + kTileSize - 1) / kTileSize,
                    (m + kTileSize - 1) / kTileSize);
  matrix_multiply_kernel<T, kTileSize>
      <<<block_layout, thread_layout>>>(A, B, C, m, k, n);
}

template void matrix_multiply<float>(const float* A, const float* B, float* C,
                                     int m, int k, int n);
template void matrix_multiply<double>(const double* A, const double* B,
                                      double* C, int m, int k, int n);
}  // namespace tiled_multiply
