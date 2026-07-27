# Advanced Matrix Multiplication

One FP32 GEMM, written seven times — a re-practice of the fundamentals of
kernel performance and profiling. Inspired by chapter 15 of *Programming
Massively Parallel Processors*: start from the naive kernel, then remove one
bottleneck at a time until the gap to cuBLAS closes. The final kernel runs an
8192² SGEMM in 23.9 ms vs cuBLAS' 21.4 ms on an H100 — a ~20× speedup over the
naive version.

## Kernel progression

| Kernel | File | What it adds |
|---|---|---|
| cuBLAS | `cublas_multiply.cu` | `cublasSgemm` reference |
| Basic | `basic_multiply.cu` | One thread per C element, every operand read straight from global memory |
| Tiled | `tiled_multiply.cu` | 32×32 shared-memory tiles — each block loads its operands once instead of 32 times |
| Large tiled | `large_tiled_multiply.cu` | 128×128 block tile (bK = 8); each thread accumulates an 8×8 register tile of C, raising arithmetic intensity per byte of shared traffic |
| ltiled | `ltiled_multiply.cu` | Inner loop restructured as an outer product: the k-slices of A and B are staged in registers, so 16 shared loads feed 64 FMAs instead of 2 loads per FMA |
| ltiled2 | `ltiled2_multiply.cu` | 128-bit `float4` global→shared loads, with a bounds-checked fallback when a dimension doesn't divide the tile |
| ltiled3 | `ltiled3_multiply.cu` | Double-buffered shared tiles + `cp.async` — the next tile streams in while the current one computes |
| ltiled4 | `ltiled4_multiply.cu` | Warp tiling (64×32 per warp) with A transposed in shared memory, so shared loads on both operands vectorize to LDS.128; one `__syncthreads` per k-tile |

## Results

H100 80GB (HBM3), square FP32 matrices, random inputs. Each cell is the
shifted geomean of 8 CUDA-event-timed runs after 3 warmups, in ms.

| Kernel | 1024² | 2048² | 4096² | 8192² |
|---|---:|---:|---:|---:|
| cuBLAS | 0.249 | 0.527 | 2.84 | 21.4 |
| Basic | 0.394 | 3.92 | 30.3 | 492 |
| Tiled | 0.240 | 1.86 | 15.3 | 121 |
| Large tiled | 0.179 | 0.825 | 6.35 | 49.8 |
| ltiled | 0.176 | 0.809 | 6.24 | 48.8 |
| ltiled2 | 0.181 | 0.862 | 6.69 | 53.1 |
| ltiled3 | 0.139 | 0.524 | 4.01 | 31.4 |
| ltiled4 | 0.111 | 0.387 | 3.02 | 23.9 |

ltiled2 is an honest data point: vectorizing the global loads is a slight
regression on its own, and only pays off in ltiled3, where the 16-byte
granularity becomes the vehicle for `cp.async`.

## Correctness

Every kernel handles arbitrary m, k, n — the vectorized fast paths fall back
to bounds-checked loads when a dimension doesn't divide the tile. `runner`
checks each kernel against cuBLAS (absolute tolerance 1e-4) on 12 shapes
chosen to stress the boundary handling, from 1×1×1 to 8191×4097×5121.

## Building and running

Requires the CUDA Toolkit (cuBLAS, cuRAND, CCCL), CMake ≥ 3.28, and a C++20
compiler. Kernels are built for the native architecture of the machine's GPU.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/runner
```

`runner` checks and times every kernel. For profiler work there is a second
binary that launches a single kernel once:

```sh
./build/profile <size> <basic|tiled|large-tiled|ltiled|l2tiled|l3tiled|l4tiled> [float|double]
```

Kernels are compiled with `-lineinfo`, so `ncu ./build/profile 4096 l4tiled`
maps counters back to source lines.
