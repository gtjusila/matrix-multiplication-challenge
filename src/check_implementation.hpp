#pragma once
#include <functional>
#include <string>
template <typename T>
void check_implementation(
    std::string const& name,
    std::function<void(const T*, const T*, T*, std::size_t, std::size_t,
                       std::size_t)>
        matrix_multiply_function);
