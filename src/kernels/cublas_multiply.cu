#include <cassert>

#include "cublas_multiply.hpp"
#include "cublas_v2.h"
#define CUBLAS_ASSERT(x) assert((x) == CUBLAS_STATUS_SUCCESS)
namespace cublas_multiply {

namespace {
class ScopedCublasHandle {
 public:
  ScopedCublasHandle() { CUBLAS_ASSERT(cublasCreate(&handle)); }
  ~ScopedCublasHandle() noexcept { cublasDestroy(handle); }
  ScopedCublasHandle(const ScopedCublasHandle& other) = delete;
  ScopedCublasHandle& operator=(const ScopedCublasHandle& other) = delete;

  operator cublasHandle_t() { return handle; };

 private:
  cublasHandle_t handle;
};
}  // namespace
template <typename T>
  requires std::same_as<T, float> || std::same_as<T, double>
void matrix_multiply(const T* A, const T* B, T* C, std::size_t m, std::size_t k,
                     std::size_t n) {
  ScopedCublasHandle handle;
  T alpha = 1.0;
  T beta = 0.0;
  if constexpr (std::is_same_v<T, float>) {
    CUBLAS_ASSERT(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              B, n, A, k, &beta, C, n));
  } else {
    CUBLAS_ASSERT(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              B, n, A, k, &beta, C, n));
  }
}
template void matrix_multiply<float>(const float* A, const float* B, float* C,
                                     std::size_t m, std::size_t k,
                                     std::size_t n);
template void matrix_multiply<double>(const double* A, const double* B,
                                      double* C, std::size_t m, std::size_t k,
                                      std::size_t n);

}  // namespace cublas_multiply
