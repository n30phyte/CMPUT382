add_executable(VectorAdd main.cu)
target_link_libraries(VectorAdd PRIVATE libwb)

set_target_properties(
        VectorAdd
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
