# Matrix Multiplication Challenge

This project explores the development and optimization of a CUDA matrix
multiplication kernel while revisiting the fundamentals of GPU performance
analysis and profiling. Inspired by Chapter 15 of *Programming Massively
Parallel Processors*, it begins with a naive implementation and progressively
addresses individual performance bottlenecks. On an H200, the fastest kernel
(ltiled5) completes an 8192² SGEMM in 21.0 ms, compared with 21.4 ms for
cuBLAS, and also outperforms cuBLAS at 1024² and 4096².

The final implementations use a block, warp, and thread tile hierarchy that
computes rank-1 updates from registers, pre-transpose A once in a standalone
kernel so both operands stream into shared memory without reshaping, and
software-pipeline the data movement — with `cp.async` (ltiled4/5) and with
Hopper's TMA (ltiled6).

## Kernel progression

| Kernel | File | What it adds |
|---|---|---|
| cuBLAS | `cublas_multiply.cu` | `cublasSgemm` reference |
| Basic | `basic_multiply.cu` | One thread per C element, every operand read straight from global memory |
| Tiled | `tiled_multiply.cu` | 32×32 shared-memory tiles — each block loads its operands once instead of 32 times |
| Large tiled | `large_tiled_multiply.cu` | Block/thread tile hierarchy: 128×128×8 block tile, each thread owning an 8×8 accumulator fragment of C, raising arithmetic intensity per byte of shared traffic |
| ltiled | `ltiled_multiply.cu` | k-loop restructured as rank-1 updates: per k-step the A/B slices are staged in registers and outer-product-accumulated into the fragment, so 16 shared loads feed 64 FMAs instead of 2 loads per FMA |
| ltiled2 | `ltiled2_multiply.cu` | Vectorized 128-bit (`float4`) global→shared loads; the vectorized and bounds-checked load paths are separate kernel instantiations via a compile-time template parameter, selected at launch when the dimensions divide the tile |
| ltiled3 | `ltiled3_multiply.cu` | Software pipelining of the global→shared stage: double-buffered tiles + `cp.async`, so tile i+1 streams in while tile i computes |
| ltiled4 | `ltiled4_multiply.cu` | Full block/warp/thread hierarchy (64×32 warp tiles) with a transposed A tile so both operands' rank-1 slices load as LDS.128; on the aligned fast path A is pre-transposed once by a standalone kernel (classic 32×32 padded-smem transpose) and its tiles `cp.async` straight into shared — no in-kernel reshape, one `__syncthreads` per k-tile; irregular sizes keep the in-kernel transpose |
| ltiled5 | `ltiled5_multiply.cu` | Configuration tuning of ltiled4: k-tile depth 16 instead of 8 halves the `__syncthreads` frequency per k-sweep (barrier convergence was the largest measured stall), and a 64×64-tile variant is dispatched when the 128×128 grid would underfill the GPU (at m=n=1024 the large tile yields only 64 blocks for 132 SMs); the fastest kernel |
| ltiled6 | `ltiled6_multiply.cu` | Hopper TMA path: the pre-transpose runs as a TMA-in/TMA-out kernel with 128B-swizzled tensor maps, and the GEMM double-buffers both tiles via `cp.async.bulk.tensor` + mbarrier, issued by a single thread; k-tile depth 16 to amortize the per-stage barrier cost. Measured slower than the equally-configured `cp.async` version (ltiled5) |

## Results

H200 141GB (HBM3e), square FP32 matrices, random inputs. Each cell is the
shifted geomean of 8 CUDA-event-timed runs after 3 warmups, in ms.

| Kernel | 1024² | 2048² | 4096² | 8192² |
|---|---:|---:|---:|---:|
| cuBLAS | 0.065 | 0.344 | 2.68 | 21.4 |
| Basic | 0.422 | 3.51 | 26.4 | 406 |
| Tiled | 0.251 | 1.92 | 15.7 | 124 |
| Large tiled | 0.179 | 0.820 | 6.47 | 50.6 |
| ltiled | 0.175 | 0.805 | 6.35 | 49.7 |
| ltiled2 | 0.174 | 0.785 | 6.17 | 48.5 |
| ltiled3 | 0.131 | 0.501 | 3.82 | 30.1 |
| ltiled4 | 0.103 | 0.372 | 2.83 | 22.3 |
| **ltiled5** | **0.062** | 0.350 | **2.67** | **21.0** |
| ltiled6 | 0.108 | 0.377 | 2.90 | 22.8 |

## Correctness

Every kernel handles arbitrary m, k, n — the vectorized fast paths fall back
to bounds-checked loads when a dimension doesn't divide the tile. `runner`
checks each kernel against cuBLAS (absolute tolerance 1e-4, relative tolerance
1e-5) on 12 shapes
chosen to stress the boundary handling, from 1×1×1 to 8191×4097×5121.

## Building and running

Requires the CUDA Toolkit (cuBLAS, cuRAND, CCCL), CMake ≥ 3.28, and a C++20
compiler. Kernels are built for the native architecture of the machine's GPU;
ltiled6's TMA path needs compute capability 9.0 (Hopper) or newer.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/runner
```

`runner` checks and times every kernel. For profiler work there is a second
binary that launches a single kernel once:

```sh
./build/profile <size> <cublas|basic|tiled|large-tiled|ltiled|l2tiled|l3tiled|l4tiled|l5tiled|l6tiled> [float|double]
```

Kernels are compiled with `-lineinfo`, so `ncu ./build/profile 4096 l4tiled`
maps counters back to source lines.
