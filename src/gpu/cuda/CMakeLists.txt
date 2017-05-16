set(MODULE CAGPU)
find_package(CUDA QUIET REQUIRED)
include(FindCUDA)

set(CUDA_ARCH -gencode arch=compute_52,code=sm_52 CACHE STRING "CUDA target architecture.")
set(CUDA_HOST_COMPILER_FLAGS -fPIC CACHE STRING "CUDA -Xcompiler flags.")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler ${CUDA_HOST_COMPILER_FLAGS}; -std=c++11; -gencode arch=compute_52,code=sm_52;)
set(CUDA_NVCC_FLAGS_DEBUG -G; -g; -O0;)

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
   CAIndexTable.cxx
   CAPrimaryVertexContext.cxx
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
