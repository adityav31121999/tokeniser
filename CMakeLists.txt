
# root CMakeLists.txt
cmake_minimum_required(VERSION 3.30.0 FATAL_ERROR)

# Backend selection options. Use -D<OPTION>=ON/OFF to change.
set(USE_CUDA OFF)
set(USE_OPENCL OFF)

if(USE_CUDA AND USE_OPENCL)
    message(FATAL_ERROR "USE_CUDA and USE_OPENCL options are mutually exclusive. Please enable only one.")
endif()

# Determine project languages based on options
set(PROJECT_LANGUAGES C CXX)
if(USE_CUDA)
    list(APPEND PROJECT_LANGUAGES CUDA)
endif()

# project and version
project(TOKENISE LANGUAGES ${PROJECT_LANGUAGES})
set(VERSION 0.0.0.1)
message("Project Name: ${PROJECT_NAME} | Version: ${VERSION}")
message("Tokenisation Tool for LLM training")

# language settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set MSVC runtime library for all targets.
# This avoids linker conflicts like LNK4098 when mixing static/dynamic runtimes.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# binaries for debug and release
set(OSbit 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSbit 64)
endif()
set(FullOutputDir "${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${OSbit}/${CMAKE_BUILD_TYPE}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static_libs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

# set compile definitions
if(USE_CUDA)
    message(STATUS "Using CUDA for compilation.")
    message(STATUS "Using CUDA for better efficiency on CUDA-based GPUs:)")
    # The CUDA language is enabled by the project() command.
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA version ${CUDAToolkit_VERSION}")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MY_CUDA_ARCH "86" CACHE STRING "Target CUDA Architecture(s) e.g., 75, 86, 90")
    set(CMAKE_CUDA_ARCHITECTURES ${MY_CUDA_ARCH})
    add_compile_definitions(USE_CUDA)
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

# List common source files for the executable
set(TOKENISE_SOURCES
    main.cpp
    tokenise.cpp
    embedding.cpp
    split.cpp
    corpus.cpp
    set.cpp
    group.cpp
    merge.cpp
    pairstats.cpp
)

# Conditionally add backend-specific source files to the list
if(USE_CUDA)
    list(APPEND TOKENISE_SOURCES kernel.cu)
elseif(USE_OPENCL)
    list(APPEND TOKENISE_SOURCES kernelcl.cpp)
endif()

add_executable(TOKENISE ${TOKENISE_SOURCES})

# Link libraries based on the selected backend
if(USE_OPENCL)
    target_link_libraries(TOKENISE PRIVATE OpenCL::OpenCL)
endif()
