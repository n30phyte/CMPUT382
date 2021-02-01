cmake_minimum_required(VERSION 3.17)
project(ParallelMerge CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(libwb)

add_executable(ParallelMerge main.cu)
target_link_libraries(ParallelMerge PRIVATE libwb)

set_target_properties(
        ParallelMerge
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")