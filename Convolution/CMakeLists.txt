add_executable(Convolution convolution.cu)
target_link_libraries(Convolution PRIVATE libwb)

set_target_properties(
        Convolution
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(TiledConvolution tiled.cu)
target_link_libraries(TiledConvolution PRIVATE libwb)

set_target_properties(
        TiledConvolution
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

