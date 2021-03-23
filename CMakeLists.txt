cmake_minimum_required(VERSION 3.17)
project(CMPUT382 CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(libwb)
include_directories(libwb)
add_subdirectory(VectorAdd)
add_subdirectory(Convolution)
add_subdirectory(ExclusiveScan)
add_subdirectory(Histogram)
add_subdirectory(MatrixMultiplication)
add_subdirectory(ParallelMerge)
add_subdirectory(Reduction)
add_subdirectory(RadixSort)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
