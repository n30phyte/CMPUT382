add_executable(Histogram main.cu)
target_link_libraries(Histogram PRIVATE libwb)

set_target_properties(
        Histogram
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
