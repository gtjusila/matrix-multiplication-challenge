#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "helpers.hpp"
#include "run_and_time_algorithm.hpp"

namespace {
class ScopedCudaEvent {
 public:
  ScopedCudaEvent() : event{} { CUDA_ASSERT(cudaEventCreate(&event)); };
  ~ScopedCudaEvent() noexcept { cudaEventDestroy(event); }
  ScopedCudaEvent(const ScopedCudaEvent& other) = delete;
  ScopedCudaEvent& operator=(const ScopedCudaEvent& other) = delete;
  ScopedCudaEvent(ScopedCudaEvent&& other) = delete;
  ScopedCudaEvent& operator=(ScopedCudaEvent&& other) = delete;

  operator cudaEvent_t() const noexcept { return event; };

 private:
  cudaEvent_t event;
};

template <typename T>
T shifted_geomean(const std::vector<T>& v, T s) {
  T log_sum = std::transform_reduce(v.begin(), v.end(), static_cast<T>(0.0),
                                    std::plus<>{},
                                    [s](T x) { return std::log(x + s); });
  return std::exp(log_sum / v.size()) - s;
}

template <typename T>
void run_and_time_algorithm_single_size(
    MultFuncType<T> matrix_multiply_function, int n) {
  ScopedCurandGenerator gen(42ULL);
  auto A = create_random_n_vector<T>(n * n, gen);
  auto B = create_random_n_vector<T>(n * n, gen);
  auto C = allocate_n_vector<T>(n * n);

  // Warm Up the kernel
  for (int i = 0; i < 3; ++i) {
    CUDA_ASSERT(cudaMemset(C.get(), 0, n * n * sizeof(T)));
    matrix_multiply_function(A.get(), B.get(), C.get(), n, n, n);
  }
  std::vector<float> timings;
  timings.reserve(8);
  for (int i = 0; i < 8; ++i) {
    CUDA_ASSERT(cudaMemset(C.get(), 0, n * n * sizeof(T)));
    ScopedCudaEvent start, stop;
    CUDA_ASSERT(cudaEventRecord(start, 0));
    matrix_multiply_function(A.get(), B.get(), C.get(), n, n, n);
    CUDA_ASSERT(cudaEventRecord(stop, 0));
    CUDA_ASSERT(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_ASSERT(cudaEventElapsedTime(&milliseconds, start, stop));
    timings.push_back(milliseconds);
  }
  std::cout << std::format("Size {}: {} ms", n, shifted_geomean(timings, 0.1f))
            << std::endl;
}
}  // namespace
template <typename T>
void run_and_time_algorithm(const std::string& name,
                            MultFuncType<T> matrix_multiply_function) {
  std::vector<int> sizes = {1024, 2048, 4096, 8192};
  std::cout << "Timing algorithm " << name << std::endl;
  for (auto size : sizes) {
    run_and_time_algorithm_single_size(matrix_multiply_function, size);
  }
}

template void run_and_time_algorithm<float>(const std::string& name,
                                            MultFuncType<float> funct);
template void run_and_time_algorithm<double>(const std::string& name,
                                             MultFuncType<double> funct);
