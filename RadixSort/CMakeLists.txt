add_executable(RadixSort kernel.cu)
target_link_libraries(RadixSort PRIVATE libwb)

add_executable(RadixSortCpu main.cpp)
target_link_libraries(RadixSortCpu PRIVATE wb)

set_target_properties(
        RadixSort
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
