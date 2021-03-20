add_executable(ExclusiveScan 2d.cu exclusive_scan.cu)
target_link_libraries(ExclusiveScan PRIVATE libwb)

add_executable(OneD 1d.cu exclusive_scan.cu)
target_link_libraries(OneD PRIVATE libwb)

set_target_properties(
        ExclusiveScan
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        OneD
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
