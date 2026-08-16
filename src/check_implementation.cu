#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <cuda/std/cmath>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "check_implementation.hpp"
#include "helpers.hpp"
#include "kernels/cublas_multiply.hpp"

namespace {
template <typename T>
struct equal_pred {
  T absolute_tolerance;
  T relative_tolerance;

  __device__ bool operator()(T actual, T expected) const {
    auto difference = cuda::std::abs(actual - expected);
    return difference <=
           absolute_tolerance + relative_tolerance * cuda::std::abs(expected);
  }
};

template <typename T>
bool matrix_equal(const T* A, const T* B, int m, int n,
                  T absolute_tolerance = static_cast<T>(1e-4),
                  T relative_tolerance = static_cast<T>(1e-5)) {
  return thrust::equal(thrust::device, A, A + (m * n), B,
                       equal_pred<T>{absolute_tolerance, relative_tolerance});
}

template <typename T>
bool check_implementation_single_size(MultFuncType<T> matrix_multiply_function,
                                      int m, int k, int n) {
  ScopedCurandGenerator gen(42ULL);
  auto A = create_random_n_vector<T>(m * k, gen);
  auto B = create_random_n_vector<T>(k * n, gen);
  auto C = allocate_n_vector<T>(m * n);
  auto D = allocate_n_vector<T>(m * n);
  matrix_multiply_function(A.get(), B.get(), C.get(), m, k, n);
  cublas_multiply::matrix_multiply(A.get(), B.get(), D.get(), m, k, n);
  return matrix_equal(C.get(), D.get(), m, n);
}
}  // namespace

template <typename T>
void check_implementation(std::string const& name,
                          MultFuncType<T> matrix_multiply_function) {
  std::cout << "Checking implementation of " << name << std::endl;

  std::vector<std::tuple<int, int, int>> test_cases = {
      {1, 1, 1},          {5, 2, 3},         {10, 1, 10}, {31, 31, 31},
      {1024, 1024, 1024}, {1023, 31, 1023},  {1, 10, 1},  {31, 63, 31},
      {4096, 1025, 2047}, {8191, 4097, 5121}, {136, 17, 136},
      {1000, 33, 1024}};

  auto cnt = 0u;

  for (auto [x, y, z] : test_cases) {
    bool pass =
        check_implementation_single_size(matrix_multiply_function, x, y, z);
    std::cout << std::format("Testcase #{} m {} k {} n {} : {}\n", cnt++, x, y,
                             z, pass ? "OK" : "FAIL");
  }
}

template void check_implementation<float>(
    const std::string& name, MultFuncType<float> matrix_multiply_function);
template void check_implementation<double>(
    const std::string& name, MultFuncType<double> matrix_multiply_function);
