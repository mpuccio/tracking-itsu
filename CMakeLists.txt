cmake_minimum_required(VERSION 3.0.2)
project(TRACKING-ITSU CXX)

include(CheckIncludeFileCXX)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -O2")
set(CMAKE_CXX_FLAGS_PROFILE "-pg" CACHE STRING "Flags used by the C++ compiler during profiling builds.")
set(CMAKE_CXX_FLAGS_MEMORYBENCHMARK "-DMEMORY_BENCHMARK" CACHE STRING "Flags used by the C++ compiler during memory benchmark builds.")

MARK_AS_ADVANCED(CMAKE_CXX_FLAGS_PROFILE)

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profile MemoryBenchmark.")

check_include_file_cxx(valgrind/callgrind.h HAVE_VALGRIND)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)
add_subdirectory(src)

add_executable(tracking-itsu-main main.cpp)
target_link_libraries(tracking-itsu-main src)