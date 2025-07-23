
# src/neural/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(nn C CXX)

add_library(nn STATIC
    nn.cpp
)

# Make the include directory available to this target and any
# other target that links to it (like the main executable).
target_include_directories(nn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# link libraries to this library
target_link_libraries(nn
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
)
