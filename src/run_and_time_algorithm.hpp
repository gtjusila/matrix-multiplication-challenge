#pragma once

#include "helpers.hpp"

template <typename T>
void run_and_time_algorithm(std::string const& name, MultFuncType<T> func);
