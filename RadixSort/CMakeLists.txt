add_executable(RadixSort kernel.cu)
target_link_libraries(RadixSort PRIVATE libwb)

add_executable(SortCpu main.cpp)
target_link_libraries(SortCpu PRIVATE libwb)

set_target_properties(
        RadixSort
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
