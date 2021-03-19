add_executable(Reduction main.cu)
target_link_libraries(Reduction PRIVATE libwb)

set_target_properties(
        Reduction
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
