set(MODULE CAGPU)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_52,code=sm_52")
set(CUDA_NVCC_FLAGS_DEBUG "-G -g -O0")

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(
    ${TRACKING-ITSU_SOURCE_DIR}/include
    ${TRACKING-ITSU_SOURCE_DIR}/include/gpu
)

set(SRCS
    CAGPUUtils.cu
)

cuda_add_library(${MODULE} ${SRCS})