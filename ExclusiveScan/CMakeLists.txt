add_executable(ExclusiveScan main.cu)
target_link_libraries(ExclusiveScan PRIVATE libwb)

set_target_properties(
        ExclusiveScan
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
