cmake_minimum_required(VERSION 2.8.0)
project(TRACKING-ITSU CXX)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Ofast")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g")

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)
add_subdirectory(src)