#include <type_traits>

#include "helper.cuh"
#include "ltiled2_multiply.hpp"

namespace ltiled2_multiply {

static constexpr int kBM = 128;
static constexpr int kBK = 8;
static constexpr int kBN = 128;
static constexpr int kTM = 8;
static constexpr int kTN = 8;
static constexpr int kThreadPerBlock = 256;

template <typename T, int sMatLd, int sMatRow, int sMatCol,
          int THREAD_PER_BLOCK>
__device__ __forceinline__ void load_tile_slow(const T* __restrict__ mat,
                                               int mat_ld, int matRow,
                                               int matCol,
                                               T* __restrict__ smat) {
  // Load sMatRow * sMatCol elements to sMat with threadPerBlock
  // separate SMatLd since sMat might be 'wider' to avoid bank conflict
  static_assert((sMatRow * sMatCol) % THREAD_PER_BLOCK == 0);
  static_assert(sMatCol <= sMatLd);
  constexpr int iter_count = (sMatRow * sMatCol) / THREAD_PER_BLOCK;
#pragma unroll
  for (int i = 0; i < iter_count; ++i) {
    int idx = (threadIdx.x + i * THREAD_PER_BLOCK);
    auto row = idx / sMatCol;
    auto col = idx % sMatCol;
    if (row < matRow && col < matCol) {
      smat[ID2X(row, col, sMatLd)] = mat[ID2X(row, col, mat_ld)];
    } else {
      smat[ID2X(row, col, sMatLd)] = 0;
    }
  }
};

template <int sMatLd, int sMatRow, int sMatCol, int THREAD_PER_BLOCK>
__device__ __forceinline__ void load_tile_fast(
    const float* __restrict__ mat, int mat_ld, float* __restrict__ smat) {
  // Each thread copies four floats with one 16-byte vectorized load/store.
  constexpr int iter_count =
      (sMatRow * sMatCol) / (THREAD_PER_BLOCK * 4);
#pragma unroll
  for (int i = 0; i < iter_count; ++i) {
    int idx = (threadIdx.x + i * THREAD_PER_BLOCK) * 4;
    auto row = idx / sMatCol;
    auto col = idx % sMatCol;
    reinterpret_cast<float4*>(&smat[ID2X(row, col, sMatLd)])[0] =
        reinterpret_cast<const float4*>(&mat[ID2X(row, col, mat_ld)])[0];
  }
};

template <typename T, int sMatLd, int sMatRow, int sMatCol,
          int THREAD_PER_BLOCK, bool FAST>
__device__ __forceinline__ void load_tile(const T* __restrict__ mat, int mat_ld,
                                          int matRow, int matCol,
                                          T* __restrict__ smat) {
  if constexpr (FAST && std::is_same_v<T, float>) {
    load_tile_fast<sMatLd, sMatRow, sMatCol, THREAD_PER_BLOCK>(mat, mat_ld,
                                                               smat);
  } else {
    load_tile_slow<T, sMatLd, sMatRow, sMatCol, THREAD_PER_BLOCK>(
        mat, mat_ld, matRow, matCol, smat);
  }
};
template <typename T, int bK, int tM, int tN>
__device__ __forceinline__ void mm(const T* __restrict__ A, int lda,
                                   const T* __restrict__ B, int ldb,
                                   T* __restrict__ C) {
#pragma unroll
  for (int j = 0; j < bK; ++j) {
    T a_r[tM];
#pragma unroll
    for (int i = 0; i < tM; ++i) {
      a_r[i] = A[ID2X(i, j, lda)];
    }
    T b_r[tN];
#pragma unroll
    for (int i = 0; i < tN; ++i) {
      b_r[i] = B[ID2X(j, i, ldb)];
    }
#pragma unroll
    for (int i = 0; i < tM * tN; ++i) {
      auto col = i % tN;
      auto row = i / tN;
      C[ID2X(row, col, tN)] += a_r[row] * b_r[col];
    }
  }
}

template <typename T, int tM, int tN>
__device__ __forceinline__ void write_tile(const T* __restrict C_r,
                                           T* __restrict__ C, int ldc, int Crow,
                                           int Ccol) {
#pragma unroll
  for (int i = 0; i < tM * tN; ++i) {
    auto col = i % tN;
    auto row = i / tN;
    if (col < Ccol && row < Crow) {
      C[ID2X(row, col, ldc)] = C_r[ID2X(row, col, tN)];
    }
  }
}
template <typename T, int bM, int bK, int bN, int tM, int tN,
          int THREAD_PER_BLOCK, bool FAST>
__global__ void __launch_bounds__(256, 1)
    matrix_multiply_kernel(const T* __restrict__ A,
                           const T* __restrict__ B, T* __restrict__ C, int m,
                           int k, int n) {
  static_assert(bM % tM == 0);
  static_assert(bN % tN == 0);
  static_assert(THREAD_PER_BLOCK == (bM * bN) / (tM * tN));

  // Both tiles unpadded: leading-dim padding does not remove the dominant b_s
  // shared-load bank conflicts (all lanes read the same k per step, so padding
  // shifts them equally) and it breaks the 16B alignment LDS.128 needs.
  __shared__ T a_s[bM * bK];
  __shared__ T b_s[bK * bN];
  T C_r[tM * tN] = {};

  int bCol = blockIdx.x * bN;
  int bRow = blockIdx.y * bM;

  int threadPerRow = bN / tN;
  int tCol = (threadIdx.x % threadPerRow) * tN;
  int tRow = (threadIdx.x / threadPerRow) * tM;

  int tile_count = (k + bK - 1) / bK;

  for (int i = 0; i < tile_count; ++i) {
    // Computing tile (y, x) of C  for each i collectively get tile (y, i) from
    // A and (i, x) from b;
    load_tile<T, bK, bM, bK, THREAD_PER_BLOCK, FAST>(
        &A[ID2X(bRow, i * bK, k)], k, m - bRow, k - (i * bK), a_s);

    load_tile<T, bN, bK, bN, THREAD_PER_BLOCK, FAST>(
        &B[ID2X(i * bK, bCol, n)], n, k - (i * bK), n - bCol, b_s);
    __syncthreads();

    mm<T, bK, tM, tN>(&a_s[ID2X(tRow, 0, bK)], bK, &b_s[ID2X(0, tCol, bN)], bN,
                      C_r);
    __syncthreads();
  }
  write_tile<T, tM, tN>(C_r, &C[ID2X(bRow + tRow, bCol + tCol, n)], n,
                        m - (bRow + tRow), n - (bCol + tCol));
};

template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n) {
  dim3 block_layout((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  // The fast path has no bounds check, so the whole tile must be in range:
  // full bM rows of A / bN cols of B, and k an exact multiple of bK.
  bool fast = (m % kBM == 0) && (n % kBN == 0) && (k % kBK == 0) &&
              (kBK % 4 == 0) && (kBN % 4 == 0);
  if (fast) {
    matrix_multiply_kernel<T, kBM, kBK, kBN, kTM, kTN, kThreadPerBlock, true>
        <<<block_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
  } else {
    matrix_multiply_kernel<T, kBM, kBK, kBN, kTM, kTN, kThreadPerBlock, false>
        <<<block_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
  }
}

template void matrix_multiply<float>(const float* A, const float* B, float* C,
                                     int m, int k, int n);
template void matrix_multiply<double>(const double* A, const double* B,
                                      double* C, int m, int k, int n);
}  // namespace ltiled2_multiply
