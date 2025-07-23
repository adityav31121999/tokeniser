
# src/neural/CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)
project(token C CXX)

add_library(token STATIC
    tokenise.cpp
    embedding.cpp
    split.cpp
    corpus.cpp
    set.cpp
    group.cpp
    merge.cpp
    pairstats.cpp
    readFiles.cpp
    train.cpp
    kernel.cu
    kernelcl.cpp
    utility.cpp
)

# Make the include directory available to this target and any
# other target that links to it (like the main executable).
target_include_directories(token PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# link libraries to this library
target_link_libraries(token
    PUBLIC # Important: Make OpenCL and CUDA linking public
        ${OpenCL_LIBRARIES}
        ${CUDAToolkit_LIBRARIES}
    nn
)
