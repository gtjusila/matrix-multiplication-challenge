#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <cstdlib>
#include <cuda/std/cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "check_implementation.hpp"
#include "cuda_helper.hpp"
#include "kernels/basic_multiply.hpp"

#define PRINT_ALGO_RES(x)          \
  do {                             \
    std::cout << (x) << std::endl; \
    std::cout << "A\n";            \
    print_matrix(A.get(), m, k);   \
    std::cout << "B\n";            \
    print_matrix(B.get(), k, n);   \
    std::cout << "C\n";            \
    print_matrix(C.get(), m, n);   \
  } while (false)

template <typename T>
void print_matrix(const T* A, std::size_t m, std::size_t n) {
  std::vector<T> temp(m * n);
  CUDA_ASSERT(
      cudaMemcpy(temp.data(), A, m * n * sizeof(T), cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (j > 0) std::cout << " ";
      std::cout << temp[i * n + j];
    }
    std::cout << std::endl;
  }
}

int main() {
  check_implementation<float>("Basic Multiply",
                              basic_multiply::matrix_multiply<float>);
}
