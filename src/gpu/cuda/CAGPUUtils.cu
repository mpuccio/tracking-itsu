/// \file CAGPUtils.cu
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

#include "CAGPUUtils.h"

#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

namespace {
void checkCUDAError(const cudaError_t error, const int line)
{
  if (error != cudaSuccess) {

    std::ostringstream errorString{};

    errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << "), line(" << line
        << ")" << std::endl;

    throw std::runtime_error{ errorString.str() };
  }
}
}

void CAGPUUtils::gpuMalloc(void **p, const int size)
{
  checkCUDAError(cudaMalloc(p, size), __LINE__);
}

void CAGPUUtils::gpuFree(void *p)
{
  checkCUDAError(cudaFree(p), __LINE__);
}

void CAGPUUtils::gpuMemset(void *p, int value, int size)
{
  checkCUDAError(cudaMemset(p, value, size), __LINE__);
}

void CAGPUUtils::gpuMemcpyHostToDevice(void *dst, const void *src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), __LINE__);
}

void CAGPUUtils::gpuStartProfiler()
{
  checkCUDAError(cudaProfilerStart(), __LINE__);
}

void CAGPUUtils::gpuStopProfiler()
{
  checkCUDAError(cudaProfilerStop(), __LINE__);
}
