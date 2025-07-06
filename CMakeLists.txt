
# root CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

# project and version
project(TOKENISE C CXX)
set(VERSION 0.0.0.1)
message("Project Name: ${PROJECT_NAME} | Version: ${VERSION}")
message("Tokenisation Tool for LLM training")

# language settings
enable_language(C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# binaries for debug and release
set(OSbit 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSbit 64)
endif()
set(FullOutputDir "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static_libs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

set(USE_OPENCL OFF)
set(USE_CUDA OFF)
# set(USE_CPU OFF)

# set compile definitions
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
    message(STATUS "Using CUDA for compilation.")
    message(STATUS "Using CUDA for better efficiency on CUDA-based GPUs:)")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MY_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture(s) e.g., 75, 86, 90")
    set(CMAKE_CUDA_ARCHITECTURES ${MY_CUDA_ARCH})
    add_compile_definitions(USE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS}) # Ensure CUDA headers are available
    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        # Ensure this property is also set for static library targets in their CMakeLists.txt files
    endif()
elseif(USE_OPENCL)
    set(OpenCL_CL_VERSION "200")
    set(CL_TARGET_OPENCL_VERSION "200")
    find_package(OpenCL REQUIRED)
    message(STATUS "Found OpenCL version ${OpenCL_CL_VERSION}")
    message(STATUS "Using OpenCL for compilation.")
    message(STATUS "Using OpenCL for better efficiency on non-CUDA CL-conformant GPUs:)")
    include_directories(${OpenCL_INCLUDE_DIRS})
    add_compile_definitions(USE_OPENCL)
else()
    add_compile_definitions(USE_CPU)
endif()

# List ALL source files needed for the executable
# Define the executable with its main source file
add_executable(TOKENISE main.cpp 
                        tokenise.cpp 
                        embedding.cpp 
                        split.cpp
                        corpus.cpp
                        support.cpp
                        set.cpp
                        kernel.cu
                        kernelcl.cpp
                        group.cpp
                        merge.cpp
                        pairstats.cpp
)


# Explicitly set runtime library for the executable target (redundant if global variable works, but safer)
if(USE_CUDA AND MSVC)
    set_target_properties(TOKENISE PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Link libraries based on the selected backend
if(USE_OPENCL)
    target_link_libraries(TOKENISE PRIVATE OpenCL::OpenCL)
endif()

