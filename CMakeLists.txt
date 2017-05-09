cmake_minimum_required(VERSION 3.0.2)
project(TRACKING-ITSU CXX)

include(CheckIncludeFileCXX)

set(TRACKINGITSU_TARGET_DEVICE CPU CACHE STRING "Target device where code must be run. Options are: CPU (default), GPU_CUDA")
set_property(CACHE TRACKINGITSU_TARGET_DEVICE PROPERTY STRINGS CPU GPU_CUDA)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -O2")
set(CMAKE_CXX_FLAGS_PROFILE "-pg" CACHE STRING "Flags used by the C++ compiler during profiling builds.")
set(CMAKE_CXX_FLAGS_MEMORYBENCHMARK -DMEMORY_BENCHMARK CACHE STRING "Flags used by the C++ compiler during memory benchmark builds.")

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_PROFILE)
MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_MEMORYBENCHMARK)

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profile MemoryBenchmark.")

check_include_file_cxx(valgrind/callgrind.h HAVE_VALGRIND)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
    message("-- Compiling for NVIDIA CUDA enabled devices")
    
    set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_PROFILE} -DGPU_PROFILING_MODE")
    
    add_definitions(-DTRACKINGITSU_CUDA_COMPILE)
    include_directories(${TRACKING-ITSU_SOURCE_DIR}/include/gpu)
    add_subdirectory(src/gpu/cuda)
endif(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

add_subdirectory(src)

add_executable(tracking-itsu-main main.cpp)
target_link_libraries(tracking-itsu-main src)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
    target_link_libraries(tracking-itsu-main CAGPU)
endif(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
