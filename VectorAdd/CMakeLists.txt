include(GoogleTest)

add_executable(VectorAdd main.cu kernel.cu)
target_link_libraries(VectorAdd PRIVATE libwb)

gtest_add_tests(VectorAddTest
        SOURCES test.cpp kernel.cu)

set_target_properties(
        VectorAdd
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
