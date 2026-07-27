#include <cuda_pipeline.h>

#include "helper.cuh"
#include "ltiled3_multiply.hpp"

namespace ltiled3_multiply {

static constexpr int kBM = 128;
static constexpr int kBK = 8;
static constexpr int kBN = 128;
static constexpr int kTM = 8;
static constexpr int kTN = 8;
static constexpr int kThreadPerBlock = 256;

template <typename T, int sMatLd, int sMatRow, int sMatCol,
          int THREAD_PER_BLOCK>
__device__ __forceinline__ void load_tile(const T* __restrict__ mat, int mat_ld,
                                          int matRow, int matCol,
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

template <typename T, int sMatLd, int sMatRow, int sMatCol,
          int THREAD_PER_BLOCK>
__device__ __forceinline__ void load_tile_fast(const T* __restrict__ mat,
                                               int mat_ld,
                                               T* __restrict__ smat) {
  // Load sMatRow * sMatCol elements to sMat with threadPerBlock threads
  // use vectorized load, that this can be done cleanly should be guaranteed by
  // caller
  static_assert((sMatRow * sMatCol) % THREAD_PER_BLOCK == 0);
  static_assert(sMatCol <= sMatLd);
  constexpr int iter_count = (sMatRow * sMatCol) / (THREAD_PER_BLOCK * 4);
#pragma unroll
  for (int i = 0; i < iter_count; ++i) {
    int idx = (threadIdx.x + i * THREAD_PER_BLOCK);
    idx *= 4;
    auto row = idx / sMatCol;
    auto col = idx % sMatCol;
    __pipeline_memcpy_async(&smat[ID2X(row, col, sMatLd)],
                            &mat[ID2X(row, col, mat_ld)], 4 * sizeof(T));
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
          int THREAD_PER_BLOCK>
__global__ void __launch_bounds__(256, 1)
    matrix_multiply_kernel(const T* __restrict__ A, const T* __restrict__ B,
                           T* __restrict__ C, int m, int k, int n) {
  static_assert(bM % tM == 0);
  static_assert(bN % tN == 0);
  static_assert(THREAD_PER_BLOCK == (bM * bN) / (tM * tN));

  // Double-buffered tiles; swap the pointers (not the arrays) each iteration.
  __shared__ T a_buf[2][bM * bK];
  __shared__ T b_buf[2][bK * bN];
  T* a_curr = a_buf[0];
  T* a_next = a_buf[1];
  T* b_curr = b_buf[0];
  T* b_next = b_buf[1];
  T C_r[tM * tN] = {};

  int bCol = blockIdx.x * bN;
  int bRow = blockIdx.y * bM;

  int threadPerRow = bN / tN;
  int tCol = (threadIdx.x % threadPerRow) * tN;
  int tRow = (threadIdx.x / threadPerRow) * tM;

  int tile_count = (k + bK - 1) / bK;

  bool p1 = (m % bM == 0) && (k % bK == 0);
  bool p2 = (n % bN == 0) && (k % bK == 0);

  // Load First Interation
  if (p1) {
    load_tile_fast<T, bK, bM, bK, THREAD_PER_BLOCK>(&A[ID2X(bRow, 0, k)], k,
                                                    a_curr);

  } else {
    load_tile<T, bK, bM, bK, THREAD_PER_BLOCK>(&A[ID2X(bRow, 0, k)], k,
                                               m - bRow, k, a_curr);
  }
  if (p2) {
    load_tile_fast<T, bN, bK, bN, THREAD_PER_BLOCK>(&B[ID2X(0, bCol, n)], n,
                                                    b_curr);
  } else {
    load_tile<T, bN, bK, bN, THREAD_PER_BLOCK>(&B[ID2X(0, bCol, n)], n, k,
                                               n - bCol, b_curr);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
  for (int i = 0; i < tile_count - 1; ++i) {
    // Computing tile (y, x) of C  for each i collectively get tile (y, i) from
    // A and (i, x) from b;
    if (p1) {
      load_tile_fast<T, bK, bM, bK, THREAD_PER_BLOCK>(
          &A[ID2X(bRow, (i + 1) * bK, k)], k, a_next);

    } else {
      load_tile<T, bK, bM, bK, THREAD_PER_BLOCK>(
          &A[ID2X(bRow, (i + 1) * bK, k)], k, m - bRow, k - ((i + 1) * bK),
          a_next);
    }

    if (p2) {
      load_tile_fast<T, bN, bK, bN, THREAD_PER_BLOCK>(
          &B[ID2X((i + 1) * bK, bCol, n)], n, b_next);
    } else {
      load_tile<T, bN, bK, bN, THREAD_PER_BLOCK>(
          &B[ID2X((i + 1) * bK, bCol, n)], n, k - ((i + 1) * bK), n - bCol,
          b_next);
    }

    mm<T, bK, tM, tN>(&a_curr[ID2X(tRow, 0, bK)], bK,
                      &b_curr[ID2X(0, tCol, bN)], bN, C_r);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    T* ta = a_curr;
    a_curr = a_next;
    a_next = ta;
    T* tb = b_curr;
    b_curr = b_next;
    b_next = tb;
  }
  mm<T, bK, tM, tN>(&a_curr[ID2X(tRow, 0, bK)], bK, &b_curr[ID2X(0, tCol, bN)],
                    bN, C_r);

  write_tile<T, tM, tN>(C_r, &C[ID2X(bRow + tRow, bCol + tCol, n)], n,
                        m - (bRow + tRow), n - (bCol + tCol));
};

template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n) {
  dim3 block_layout((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  matrix_multiply_kernel<T, kBM, kBK, kBN, kTM, kTN, kThreadPerBlock>
      <<<block_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
}

template void matrix_multiply<float>(const float* A, const float* B, float* C,
                                     int m, int k, int n);
template void matrix_multiply<double>(const double* A, const double* B,
                                      double* C, int m, int k, int n);
}  // namespace ltiled3_multiply
