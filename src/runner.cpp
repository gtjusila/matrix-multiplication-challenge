#include <iostream>
#include <string>

#include "check_implementation.hpp"
#include "helpers.hpp"
#include "kernels/basic_multiply.hpp"
#include "kernels/cublas_multiply.hpp"
#include "kernels/large_tiled_multiply.hpp"
#include "kernels/tiled_multiply.hpp"
#include "run_and_time_algorithm.hpp"

template <typename T>
void check_and_time(std::string const& name, MultFuncType<T> func) {
  std::cout << std::endl;
  check_implementation(name, func);
  run_and_time_algorithm(name, func);
}
int main() {
  run_and_time_algorithm<float>("CuBlas Multiply",
                                cublas_multiply::matrix_multiply<float>);

  // check_and_time<float>("Basic Multiplication",
  // basic_multiply::matrix_multiply<float>);
  check_and_time<float>("Tiled Multiplication",
                        tiled_multiply::matrix_multiply<float>);
  check_and_time<float>("Large Tiled Multiplication",
                        large_tiled_multiply::matrix_multiply<float>);
}
