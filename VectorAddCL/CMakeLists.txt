add_executable(VectorAddCL main.cpp)

target_link_libraries(VectorAddCL PRIVATE libwb)
target_link_libraries(VectorAddCL PRIVATE OpenCL::OpenCL)