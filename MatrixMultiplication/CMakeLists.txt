add_executable(NaiveGPU gpu_mul.cu)
target_link_libraries(NaiveGPU PRIVATE libwb)

add_executable(NaiveCPU cpu.cpp)
target_link_libraries(NaiveCPU PRIVATE libwb)

add_executable(OptGPU opt_mul.cu)
target_link_libraries(OptGPU PRIVATE libwb)

set_target_properties(
        NaiveGPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 35)

set_target_properties(
        NaiveCPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 35)

set_target_properties(
        OptGPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 35)

