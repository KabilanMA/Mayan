cmake_minimum_required(VERSION 3.10)
project(SparseEinsumCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(mayan src/main.cpp)