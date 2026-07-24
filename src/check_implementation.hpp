#pragma once
#include <string>

#include "helpers.hpp"

template <typename T>
void check_implementation(std::string const& name,
                          MultFuncType<T> matrix_multiply_function);
