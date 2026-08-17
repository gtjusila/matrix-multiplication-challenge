#include <cooperative_groups.h>
#include <cuda_pipeline.h>

#include <stdexcept>
#include <type_traits>

#include "helper.cuh"
#include "ltiled4_multiply.hpp"

namespace ltiled4_multiply {

static constexpr int kBM = 256;
static constexpr int kBK = 8;
static constexpr int kBN = 128;
static constexpr int kWM = 128;
static constexpr int kWN = 32;
static constexpr int kTM = 16;
static constexpr int kTN = 8;
static constexpr int kThreadPerBlock = 256;
// widen a_t's rows so the transpose's stores and mm's loads miss each other's
// banks; must stay a multiple of 4 to keep mm's loads vectorized
static constexpr int kATPad = 4;
static constexpr int kTrTile = 32;
static constexpr int kTrBlockRows = 8;

namespace cg = cooperative_groups;
using warp_group_t = decltype(cg::tiled_partition<32>(cg::this_thread_block()));

template <typename T, int sMatLd, int sMatRow, int sMatCol,
          int THREAD_PER_BLOCK>
__device__ __forceinline__ void load_tile_slow(const T* __restrict__ mat,
                                               int mat_ld, int matRow,
                                               int matCol,
                                               T* __restrict__ smat) {
  // Load sMatRow * sMatCol elements to sMat with THREAD_PER_BLOCK threads
  // same thread -> element mapping as load_tile_fast, so transpose_tile only
  // ever reads back bytes this thread itself wrote
  static_assert(sMatCol % 4 == 0, "a thread's 4 columns must stay in one row");
  constexpr int iter_count = (sMatRow * sMatCol) / (THREAD_PER_BLOCK * 4);
#pragma unroll
  for (int i = 0; i < iter_count; ++i) {
    int idx = (threadIdx.x + i * THREAD_PER_BLOCK) * 4;
    auto row = idx / sMatCol;
    auto col0 = idx % sMatCol;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      auto col = col0 + c;
      smat[ID2X(row, col, sMatLd)] =
          (row < matRow && col < matCol) ? mat[ID2X(row, col, mat_ld)] : T(0);
    }
  }
};

template <int sMatLd, int sMatRow, int sMatCol, int THREAD_PER_BLOCK>
__device__ __forceinline__ void load_tile_fast(
    const float* __restrict__ mat, int mat_ld, float* __restrict__ smat) {
  // Each thread copies four floats with one 16-byte transaction.
  constexpr int iter_count =
      (sMatRow * sMatCol) / (THREAD_PER_BLOCK * 4);

#pragma unroll
  for (int i = 0; i < iter_count; ++i) {
    int idx = (threadIdx.x + i * THREAD_PER_BLOCK) * 4;
    auto row = idx / sMatCol;
    auto col = idx % sMatCol;
    __pipeline_memcpy_async(&smat[ID2X(row, col, sMatLd)],
                            &mat[ID2X(row, col, mat_ld)], 16);
  }
};

template <typename T, int sMatLd, int sMatRow, int sMatCol,
          int THREAD_PER_BLOCK, bool FAST>
__device__ __forceinline__ void load_tile(const T* __restrict__ mat, int mat_ld,
                                          int matRow, int matCol,
                                          T* __restrict__ smat) {
  static_assert((sMatRow * sMatCol) % THREAD_PER_BLOCK == 0);
  static_assert(sMatCol <= sMatLd);
  if constexpr (FAST && std::is_same_v<T, float>) {
    load_tile_fast<sMatLd, sMatRow, sMatCol, THREAD_PER_BLOCK>(mat, mat_ld,
                                                               smat);
  } else {
    load_tile_slow<T, sMatLd, sMatRow, sMatCol, THREAD_PER_BLOCK>(
        mat, mat_ld, matRow, matCol, smat);
  }
};

template <typename T, int bM, int bK, int aTLd, int THREAD_PER_BLOCK>
__device__ __forceinline__ void transpose_tile(const T* __restrict__ src,
                                               T* __restrict__ dst) {
  // bM x bK (ld bK) -> bK x bM (ld aTLd), both in shared
  static_assert(bK % 4 == 0, "a thread's 4 columns must stay inside one row");
  static_assert((bM * bK) % (THREAD_PER_BLOCK * 4) == 0);
  constexpr int iter_count = (bM * bK) / (THREAD_PER_BLOCK * 4);

#pragma unroll
  for (int i = 0; i < iter_count; ++i) {
    // read src[idx..idx+3] flat, not via (row, col); the compiler needs to see
    // the 4-element stride to fold them into one LDS.128
    int idx = (threadIdx.x + i * THREAD_PER_BLOCK) * 4;
    auto row = idx / bK;
    auto col = idx % bK;
    T v[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      v[c] = src[idx + c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      dst[ID2X(col + c, row, aTLd)] = v[c];
    }
  }
}

// A arrives transposed, so a thread's tM rows for one j are contiguous and
// load as LDS.128 instead of striding onto the same bank
template <typename T, int bK, int wM, int wN, int tM, int tN>
__device__ __forceinline__ void mm(warp_group_t& warp, const T* __restrict__ A,
                                   int lda, const T* __restrict__ B, int ldb,
                                   T* __restrict__ C) {
  static_assert(tM % 2 == 0);
  static_assert(wM % 2 == 0);
  static_assert(wN % 2 == 0);
  static_assert(tN % 2 == 0);
  int lane = warp.thread_rank();
  constexpr auto sTM = tM / 2;
  constexpr auto sTN = tN / 2;

  auto threadPerRow = wN / tN;

  auto tColId = (lane % threadPerRow);
  auto tRowId = (lane / threadPerRow);

#pragma unroll
  for (int j = 0; j < bK; ++j) {
    T a_r[tM];
#pragma unroll
    for (int i = 0; i < sTM; ++i) {
      a_r[i] = A[ID2X(j, i + tRowId * sTM, lda)];
    }
#pragma unroll
    for (int i = 0; i < sTM; ++i) {
      a_r[i + sTM] = A[ID2X(j, i + tRowId * sTM + wM / 2, lda)];
    }
    T b_r[tN];
#pragma unroll
    for (int i = 0; i < sTN; ++i) {
      b_r[i] = B[ID2X(j, i + sTN * tColId, ldb)];
    }
#pragma unroll
    for (int i = 0; i < sTN; ++i) {
      b_r[i + sTN] = B[ID2X(j, i + sTN * tColId + wN / 2, ldb)];
    }

#pragma unroll
    for (int i = 0; i < tM * tN; ++i) {
      auto col = i % tN;
      auto row = i / tN;
      C[ID2X(row, col, tN)] += a_r[row] * b_r[col];
    }
  }
}

template <typename T, int wM, int wN, int tM, int tN>
__device__ __forceinline__ void write_tile(warp_group_t& warp,
                                           const T* __restrict__ C_r,
                                           T* __restrict__ C, int ldc, int Crow,
                                           int Ccol, bool fast_predicate) {
  static_assert(tM % 2 == 0);
  static_assert(wM % 2 == 0);
  static_assert(wN % 2 == 0);
  static_assert(tN % 2 == 0);
  int lane = warp.thread_rank();
  constexpr auto sTM = tM / 2;
  constexpr auto sTN = tN / 2;

  auto threadPerRow = wN / tN;

  auto tCol = (lane % threadPerRow) * sTN;
  auto tRow = (lane / threadPerRow) * sTM;

  constexpr int gColBase[] = {0, wN / 2, 0, wN / 2};
  constexpr int gRowBase[] = {0, 0, wM / 2, wM / 2};

  constexpr int rColBase[] = {0, tN / 2, 0, tN / 2};
  constexpr int rRowBase[] = {0, 0, tM / 2, tM / 2};

  bool fast = false;

  if constexpr (sTN == 4 && (wN / 2) % 4 == 0 && std::is_same_v<T, float>) {
    if (fast_predicate) {
      fast = true;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < sTM; ++i) {
          reinterpret_cast<float4*>(
              &C[ID2X(i + tRow + gRowBase[j], tCol + gColBase[j], ldc)])[0] =
              reinterpret_cast<const float4*>(
                  &C_r[ID2X(i + rRowBase[j], rColBase[j], tN)])[0];
        }
      }
    }
  }
  if (!fast) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < sTM * sTN; ++i) {
        int col = i % sTN;
        int row = i / sTN;
        if (col + tCol + gColBase[j] < Ccol &&
            row + tRow + gRowBase[j] < Crow) {
          C[ID2X(row + tRow + gRowBase[j], col + tCol + gColBase[j], ldc)] =
              C_r[ID2X(row + rRowBase[j], col + rColBase[j], tN)];
        }
      }
    }
  }
}

template <typename T, int bM, int bK, int bN, int aTLd, int wM, int wN, int tM,
          int tN, int THREAD_PER_BLOCK, bool FAST>
__global__ void __launch_bounds__(THREAD_PER_BLOCK, 1)
    matrix_multiply_kernel(const T* __restrict__ A, const T* __restrict__ B,
                           T* __restrict__ C, int m, int k, int n) {
  static_assert(THREAD_PER_BLOCK == (bM * bN) / (tM * tN));
  static_assert(bM % wM == 0);
  static_assert(bN % wN == 0);
  static_assert(wN % tN == 0);
  static_assert(wM % tM == 0);
  static_assert((wM / tM) * (wN / tN) == 32);
  static_assert(aTLd >= bM);

  // a_buf takes the raw cp.async tile, a_t the transposed one mm reads.
  // a_t is double buffered so the transpose never writes the buffer mm is on;
  // a_buf needs only one, each thread rereads just the bytes it wrote itself.
  __shared__ alignas(16) T a_buf[bM * bK];
  __shared__ alignas(16) T a_t_buf[2][bK * aTLd];
  __shared__ alignas(16) T b_buf[2][bK * bN];

  T* a_t_curr = a_t_buf[0];
  T* a_t_next = a_t_buf[1];
  T* b_curr = b_buf[0];
  T* b_next = b_buf[1];
  alignas(16) T C_r[tM * tN] = {};

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  int bCol = blockIdx.x * bN;
  int bRow = blockIdx.y * bM;

  int warpPerRow = bN / wN;
  int wCol = (warp.meta_group_rank() % warpPerRow) * wN;
  int wRow = (warp.meta_group_rank() / warpPerRow) * wM;

  // Load First Interation
  load_tile<T, bK, bM, bK, THREAD_PER_BLOCK, FAST>(&A[ID2X(bRow, 0, k)], k,
                                                   m - bRow, k, a_buf);
  load_tile<T, bN, bK, bN, THREAD_PER_BLOCK, FAST>(&B[ID2X(0, bCol, n)], n, k,
                                                   n - bCol, b_curr);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  // no barrier before the transpose; wait_prior covers this thread's own copies
  transpose_tile<T, bM, bK, aTLd, THREAD_PER_BLOCK>(a_buf, a_t_curr);
  __syncthreads();

  int tile_count = (k + bK - 1) / bK;
  for (int i = 0; i < tile_count - 1; ++i) {
    // Computing tile (y, x) of C  for each i collectively get tile (y, i) from
    // A and (i, x) from b; issue the copies first so they fly while mm runs
    load_tile<T, bK, bM, bK, THREAD_PER_BLOCK, FAST>(
        &A[ID2X(bRow, (i + 1) * bK, k)], k, m - bRow, k - ((i + 1) * bK),
        a_buf);

    load_tile<T, bN, bK, bN, THREAD_PER_BLOCK, FAST>(
        &B[ID2X((i + 1) * bK, bCol, n)], n, k - ((i + 1) * bK), n - bCol,
        b_next);
    __pipeline_commit();

    mm<T, bK, wM, wN, tM, tN>(warp, &a_t_curr[ID2X(0, wRow, aTLd)], aTLd,
                              &b_curr[ID2X(0, wCol, bN)], bN, C_r);

    __pipeline_wait_prior(0);
    transpose_tile<T, bM, bK, aTLd, THREAD_PER_BLOCK>(a_buf, a_t_next);

    // the only barrier; publishes a_t_next / b_next and retires the mm reads
    __syncthreads();

    T* tat = a_t_curr;
    a_t_curr = a_t_next;
    a_t_next = tat;
    T* tb = b_curr;
    b_curr = b_next;
    b_next = tb;
  }

  mm<T, bK, wM, wN, tM, tN>(warp, &a_t_curr[ID2X(0, wRow, aTLd)], aTLd,
                            &b_curr[ID2X(0, wCol, bN)], bN, C_r);

  write_tile<T, wM, wN, tM, tN>(
      warp, C_r, &C[ID2X(bRow + wRow, bCol + wCol, n)], n, m - (bRow + wRow),
      n - (bCol + wCol), (m % bM == 0) && (n % bN == 0));
};

// transposeCoalesced scheme from Mark Harris' "An Efficient Matrix Transpose
// in CUDA C/C++",
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
__global__ void __launch_bounds__(kTrTile* kTrBlockRows)
    transpose_kernel(const float* __restrict__ src, float* __restrict__ dst,
                     int m, int k) {
  __shared__ float tile[kTrTile][kTrTile + 1];

  int col = blockIdx.x * kTrTile + threadIdx.x;
  int row0 = blockIdx.y * kTrTile + threadIdx.y;
#pragma unroll
  for (int i = 0; i < kTrTile; i += kTrBlockRows) {
    tile[threadIdx.y + i][threadIdx.x] = src[ID2X(row0 + i, col, k)];
  }
  __syncthreads();

  int tCol = blockIdx.y * kTrTile + threadIdx.x;
  int tRow0 = blockIdx.x * kTrTile + threadIdx.y;
#pragma unroll
  for (int i = 0; i < kTrTile; i += kTrBlockRows) {
    dst[ID2X(tRow0 + i, tCol, m)] = tile[threadIdx.x][threadIdx.y + i];
  }
}

inline float* get_transpose_workspace(size_t count) {
  struct Workspace {
    float* ptr = nullptr;
    size_t capacity = 0;
    ~Workspace() {
      if (ptr) cudaFree(ptr);
    }
  };
  static thread_local Workspace ws;
  if (ws.capacity < count) {
    if (ws.ptr) cudaFree(ws.ptr);
    ws.ptr = nullptr;
    ws.capacity = 0;
    if (cudaMalloc(&ws.ptr, count * sizeof(float)) != cudaSuccess) {
      throw std::runtime_error("transpose workspace allocation failed");
    }
    ws.capacity = count;
  }
  return ws.ptr;
}

template <int bM, int bK, int bN, int wM, int wN, int tM, int tN,
          int THREAD_PER_BLOCK>
__global__ void __launch_bounds__(THREAD_PER_BLOCK, 1)
    matrix_multiply_kernel_pret(const float* __restrict__ At,
                                const float* __restrict__ B,
                                float* __restrict__ C, int m, int k, int n) {
  static_assert(THREAD_PER_BLOCK == (bM * bN) / (tM * tN));
  static_assert((wM / tM) * (wN / tN) == 32);

  __shared__ alignas(16) float a_t_buf[2][bK * bM];
  __shared__ alignas(16) float b_buf[2][bK * bN];

  float* a_t_curr = a_t_buf[0];
  float* a_t_next = a_t_buf[1];
  float* b_curr = b_buf[0];
  float* b_next = b_buf[1];
  alignas(16) float C_r[tM * tN] = {};

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  int bCol = blockIdx.x * bN;
  int bRow = blockIdx.y * bM;

  int warpPerRow = bN / wN;
  int wCol = (warp.meta_group_rank() % warpPerRow) * wN;
  int wRow = (warp.meta_group_rank() / warpPerRow) * wM;

  load_tile_fast<bM, bK, bM, THREAD_PER_BLOCK>(&At[ID2X(0, bRow, m)], m,
                                               a_t_curr);
  load_tile_fast<bN, bK, bN, THREAD_PER_BLOCK>(&B[ID2X(0, bCol, n)], n, b_curr);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  int tile_count = k / bK;
  for (int i = 0; i < tile_count - 1; ++i) {
    load_tile_fast<bM, bK, bM, THREAD_PER_BLOCK>(
        &At[ID2X((i + 1) * bK, bRow, m)], m, a_t_next);
    load_tile_fast<bN, bK, bN, THREAD_PER_BLOCK>(
        &B[ID2X((i + 1) * bK, bCol, n)], n, b_next);
    __pipeline_commit();

    mm<float, bK, wM, wN, tM, tN>(warp, &a_t_curr[ID2X(0, wRow, bM)], bM,
                                  &b_curr[ID2X(0, wCol, bN)], bN, C_r);

    __pipeline_wait_prior(0);
    __syncthreads();

    float* tat = a_t_curr;
    a_t_curr = a_t_next;
    a_t_next = tat;
    float* tb = b_curr;
    b_curr = b_next;
    b_next = tb;
  }

  mm<float, bK, wM, wN, tM, tN>(warp, &a_t_curr[ID2X(0, wRow, bM)], bM,
                                &b_curr[ID2X(0, wCol, bN)], bN, C_r);

  write_tile<float, wM, wN, tM, tN>(warp, C_r,
                                    &C[ID2X(bRow + wRow, bCol + wCol, n)], n,
                                    wM, wN, true);
};

template <typename T>
void matrix_multiply(const T* A, const T* B, T* C, int m, int k, int n) {
  if constexpr (std::is_same_v<T, double>) {
    // 256-wide shared buffers exceed the 48KB static limit with 8B elements;
    // keep double on the 128x128 shape
    constexpr int dBM = 128, dWM = 64, dTM = 8;
    dim3 double_layout((n + kBN - 1) / kBN, (m + dBM - 1) / dBM);
    bool dfast = (m % dBM == 0) && (n % kBN == 0) && (k % kBK == 0);
    if (dfast) {
      matrix_multiply_kernel<T, dBM, kBK, kBN, dBM + kATPad, dWM, kWN, dTM,
                             kTN, kThreadPerBlock, true>
          <<<double_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
    } else {
      matrix_multiply_kernel<T, dBM, kBK, kBN, dBM + kATPad, dWM, kWN, dTM,
                             kTN, kThreadPerBlock, false>
          <<<double_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
    }
    return;
  } else {
    dim3 block_layout((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
    bool fast = (m % kBM == 0) && (n % kBN == 0) && (k % kBK == 0) &&
                (kBK % 4 == 0) && (kBN % 4 == 0);
    if (fast && (k % kTrTile == 0)) {
      float* At = get_transpose_workspace(size_t(m) * k);
      dim3 transpose_layout(k / kTrTile, m / kTrTile);
      transpose_kernel<<<transpose_layout, dim3(kTrTile, kTrBlockRows)>>>(
          A, At, m, k);
      matrix_multiply_kernel_pret<kBM, kBK, kBN, kWM, kWN, kTM, kTN,
                                  kThreadPerBlock>
          <<<block_layout, kThreadPerBlock>>>(At, B, C, m, k, n);
      return;
    }
    if (fast) {
      matrix_multiply_kernel<T, kBM, kBK, kBN, kBM + kATPad, kWM, kWN, kTM,
                             kTN, kThreadPerBlock, true>
          <<<block_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
    } else {
      matrix_multiply_kernel<T, kBM, kBK, kBN, kBM + kATPad, kWM, kWN, kTM,
                             kTN, kThreadPerBlock, false>
          <<<block_layout, kThreadPerBlock>>>(A, B, C, m, k, n);
    }
  }
}

template void matrix_multiply<float>(const float* A, const float* B, float* C,
                                     int m, int k, int n);
template void matrix_multiply<double>(const double* A, const double* B,
                                      double* C, int m, int k, int n);
}  // namespace ltiled4_multiply
