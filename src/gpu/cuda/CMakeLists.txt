set(MODULE CAGPU)
find_package(CUDA QUIET REQUIRED)
include(FindCUDA)

set(CUDA_ARCH -gencode arch=compute_52,code=sm_52 CACHE STRING "CUDA target architecture.")
set(CUDA_HOST_COMPILER_FLAGS -fPIC CACHE STRING "CUDA -Xcompiler flags.")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler ${CUDA_HOST_COMPILER_FLAGS}; -std=c++11; --maxrregcount=64; -gencode arch=compute_52,code=sm_52;)

# Cannot use CUDA_NVCC_FLAGS_<CONFIG> due to issue 16411
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -G; -g; -O2; -DDEBUG;)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Profile")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; --use_fast_math;)
endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Profile")    

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")

include_directories(
    ${TRACKING-ITSU_SOURCE_DIR}/include
    ${TRACKING-ITSU_SOURCE_DIR}/include/gpu
)

set(CUDA_SRCS
    CAGPUUtils.cu
    CAGPUTrackingAPI.cu
)

set(DEPENDENT_HOST_SRCS
   CATrackingUtils.cxx
   CATracklet.cxx
)

foreach(HOST_SRC_FILE ${DEPENDENT_HOST_SRCS})
    
    get_filename_component(HOST_SRC_FILENAME ${HOST_SRC_FILE} NAME_WE)
    
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HOST_SRC_FILENAME}.cu
        COMMAND ${CMAKE_COMMAND} -E copy ${TRACKING-ITSU_SOURCE_DIR}/src/${HOST_SRC_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${HOST_SRC_FILENAME}.cu
        DEPENDS ${TRACKING-ITSU_SOURCE_DIR}/src/${HOST_SRC_FILE}
    )
    
    set(CUDA_SRCS ${CUDA_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/${HOST_SRC_FILENAME}.cu)
    
endforeach(HOST_SRC_FILE)

cuda_add_library(${MODULE} ${CUDA_SRCS})
