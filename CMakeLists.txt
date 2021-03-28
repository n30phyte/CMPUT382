cmake_minimum_required(VERSION 3.17)
project(CMPUT382 CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(external)
include_directories(external/libwb)

add_subdirectory(VectorAdd)            # Lab 2
add_subdirectory(ParallelMerge)        # Lab 3
add_subdirectory(MatrixMultiplication) # Lab 4
add_subdirectory(Histogram)            # Lab 5
add_subdirectory(Convolution)          # Lab 6
add_subdirectory(Reduction)            # Lab 7
add_subdirectory(ExclusiveScan)        # Lab 8
add_subdirectory(RadixSort)            # Lab 9

find_package(OpenCL REQUIRED)

add_subdirectory(VectorAddCL)          # Lab 10

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
