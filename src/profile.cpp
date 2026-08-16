#include <cstring>
#include <iostream>
#include <string>

#include "helpers.hpp"
#include "kernels/basic_multiply.hpp"
#include "kernels/large_tiled_multiply.hpp"
#include "kernels/ltiled2_multiply.hpp"
#include "kernels/ltiled3_multiply.hpp"
#include "kernels/ltiled4_multiply.hpp"
#include "kernels/ltiled5_multiply.hpp"
#include "kernels/ltiled_multiply.hpp"
#include "kernels/tiled_multiply.hpp"

#define REGISTER_ALGORITHM(alg, func)                            \
  do {                                                           \
    registered_algorithm.push_back(alg);                         \
    if (std::strcmp(algorithm.c_str(), alg) == 0 && argc >= 2) { \
      if (use_float) {                                           \
        profile_mult_function<float>(n, func<float>);            \
      } else {                                                   \
        profile_mult_function<double>(n, func<double>);          \
      }                                                          \
    }                                                            \
  } while (false)

template <typename T>
void profile_mult_function(int n, MultFuncType<T> matrix_multiply_function) {
  ScopedCurandGenerator gen(42ULL);
  auto A = create_random_n_vector<T>(n * n, gen);
  auto B = create_random_n_vector<T>(n * n, gen);
  auto C = allocate_n_vector<T>(n * n);
  matrix_multiply_function(A.get(), B.get(), C.get(), n, n, n);
};

int main(int argc, const char* argv[]) {
  int n = 4096;
  std::vector<std::string> registered_algorithm;
  if (argc >= 2) {
    n = std::stoi(argv[1]);
  }

  std::string algorithm = "l-tiled";
  if (argc >= 3) {
    algorithm = argv[2];
  }
  bool use_float = true;
  if (argc >= 4 && (std::strcmp(argv[3], "double") == 0)) {
    use_float = false;
  }

  REGISTER_ALGORITHM("basic", basic_multiply::matrix_multiply);
  REGISTER_ALGORITHM("tiled", tiled_multiply::matrix_multiply);
  REGISTER_ALGORITHM("large-tiled", large_tiled_multiply::matrix_multiply);
  REGISTER_ALGORITHM("ltiled", ltiled_multiply::matrix_multiply);
  REGISTER_ALGORITHM("l2tiled", ltiled2_multiply::matrix_multiply);
  REGISTER_ALGORITHM("l3tiled", ltiled3_multiply::matrix_multiply);
  REGISTER_ALGORITHM("l4tiled", ltiled4_multiply::matrix_multiply);
  REGISTER_ALGORITHM("l5tiled", ltiled5_multiply::matrix_multiply);

  if (argc == 1) {
    std::string p;
    for (int i = 0; i < registered_algorithm.size(); ++i) {
      if (i != 0) p += "/";
      p += registered_algorithm[i];
    }
    std::cout << std::format(
                     "./profile <size> <algorithm: {}> <empty/float/double>", p)
              << std::endl;
  }
}
