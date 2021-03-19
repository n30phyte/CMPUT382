add_executable(ParallelMerge main.cu)
target_link_libraries(ParallelMerge PRIVATE libwb)

set_target_properties(
        ParallelMerge
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
