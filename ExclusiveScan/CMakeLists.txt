add_executable(ExclusiveScan 2d.cu exclusive_scan.cu)
target_link_libraries(ExclusiveScan PRIVATE libwb)

add_executable(ExclusiveScan1D 1d.cu exclusive_scan.cu)
target_link_libraries(ExclusiveScan1D PRIVATE libwb)

set_target_properties(
        ExclusiveScan
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        ExclusiveScan1D
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
