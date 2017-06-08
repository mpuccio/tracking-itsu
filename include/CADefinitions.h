/// \file CADefinitions.h
/// \brief
///
/// \author Iacopo Colonnelli, Politecnico di Torino
///
/// \copyright Copyright (C) 2017  Iacopo Colonnelli. \n\n
///   This program is free software: you can redistribute it and/or modify
///   it under the terms of the GNU General Public License as published by
///   the Free Software Foundation, either version 3 of the License, or
///   (at your option) any later version. \n\n
///   This program is distributed in the hope that it will be useful,
///   but WITHOUT ANY WARRANTY; without even the implied warranty of
///   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///   GNU General Public License for more details. \n\n
///   You should have received a copy of the GNU General Public License
///   along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#ifndef TRACKINGITSU_INCLUDE_CADEFINITIONS_H_
#define TRACKINGITSU_INCLUDE_CADEFINITIONS_H_

#if defined(TRACKINGITSU_CUDA_COMPILE)
# define TRACKINGITSU_GPU_MODE true
#else
# define TRACKINGITSU_GPU_MODE false
#endif

#if defined(__CUDACC__)
# define TRACKINGITSU_GPU_COMPILING
#endif

#define TRACKINGITSU_HOST_COMPILATION_NAMESPACE HostCompiledCode
#define TRACKINGITSU_DEVICE_COMPILATION_NAMESPACE DeviceCompiledCode

#if defined(TRACKINGITSU_GPU_COMPILING)
# define TRACKINGITSU_TARGET_NAMESPACE TRACKINGITSU_DEVICE_COMPILATION_NAMESPACE
#else
# define TRACKINGITSU_TARGET_NAMESPACE TRACKINGITSU_HOST_COMPILATION_NAMESPACE
#endif

#if defined(__CUDACC__)

# define GPU_HOST __host__
# define GPU_DEVICE __device__
# define GPU_HOST_DEVICE __host__ __device__
# define GPU_GLOBAL __global__
# define GPU_SHARED __shared__
# define GPU_SYNC __syncthreads()

# define MATH_ABS abs
# define MATH_ATAN2 atan2
# define MATH_MAX max
# define MATH_MIN min
# define MATH_SQRT sqrt

# include "CAGPUArray.h"
# define GPU_ARRAY CAGPUArray

#else

# define GPU_HOST
# define GPU_DEVICE
# define GPU_HOST_DEVICE
# define GPU_GLOBAL
# define GPU_SHARED
# define GPU_SYNC

# define MATH_ABS std::abs
# define MATH_ATAN2 std::atan2
# define MATH_MAX std::max
# define MATH_MIN std::min
# define MATH_SQRT std::sqrt

typedef struct _float2 { float x; float y; } float2;
typedef struct _float3 { float x; float y; float z; } float3;

# include <array>
# define GPU_ARRAY std::array

#endif

#endif /* TRACKINGITSU_INCLUDE_CADEFINITIONS_H_ */
