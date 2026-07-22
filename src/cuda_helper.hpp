#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP
#include <cassert>
#include <concepts>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>

#include "cuda_runtime.h"
#include "curand.h"
#define CUDA_ASSERT(x)                                                 \
  do {                                                                 \
    if (auto err = (x); err != cudaSuccess) {                          \
      std::cout << std::format("Error at {}:{} -- {}: {}\n", __FILE__, \
                               __LINE__, cudaGetErrorName(err),        \
                               cudaGetErrorString(err));               \
      assert(false);                                                   \
    }                                                                  \
  } while (false)

#define CURAND_ASSERT(x)                                                \
  do {                                                                  \
    if ((x) != CURAND_STATUS_SUCCESS) {                                 \
      std::cout << std::format("Error at {}:{}\n", __FILE__, __LINE__); \
      assert(false);                                                    \
    }                                                                   \
  } while (false)

template <typename T>
struct CudaDeleter {
  void operator()(T* ptr) noexcept {
    if (ptr) CUDA_ASSERT(cudaFree(static_cast<void*>(ptr)));
  }
};

template <typename T>
using CuPtr = std::unique_ptr<T, CudaDeleter<T>>;
class ScopedCurandGenerator {
 public:
  ScopedCurandGenerator() : gen{} {
    CURAND_ASSERT(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_ASSERT(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));
  }
  ScopedCurandGenerator(unsigned long long seed) : gen{} {
    CURAND_ASSERT(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_ASSERT(curandSetPseudoRandomGeneratorSeed(gen, seed));
  }

  ~ScopedCurandGenerator() { CURAND_ASSERT(curandDestroyGenerator(gen)); }
  ScopedCurandGenerator(const ScopedCurandGenerator&) = delete;
  ScopedCurandGenerator& operator=(const ScopedCurandGenerator&) = delete;
  operator curandGenerator_t() const noexcept { return gen; };

 private:
  curandGenerator_t gen;
};

template <typename T>
auto allocate_n_vector(std::size_t n) {
  T* temp;
  CUDA_ASSERT(cudaMalloc(&temp, n * sizeof(T)));
  return CuPtr<T>(temp);
}

template <typename T>
  requires std::same_as<T, float> || std::same_as<T, double>
auto create_random_n_vector(std::size_t n, curandGenerator_t gen) {
  auto temp = allocate_n_vector<T>(n);
  if constexpr (std::is_same_v<T, float>) {
    CURAND_ASSERT(curandGenerateUniform(gen, temp.get(), n));
  } else {
    CURAND_ASSERT(curandGenearteUniformDouble(gen, temp.get(), n));
  }
  return temp;
}

#endif
