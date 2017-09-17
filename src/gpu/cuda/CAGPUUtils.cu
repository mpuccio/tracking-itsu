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

#include "CAGPUContext.h"

namespace {
void checkCUDAError(const cudaError_t error, const char *file, const int line)
{
  if (error != cudaSuccess) {

    std::ostringstream errorString { };

    errorString << file << ":" << line << " CUDA API returned error [" << cudaGetErrorString(error) << "] (code "
        << error << ")" << std::endl;

    throw std::runtime_error { errorString.str() };
  }
}

int roundUp(const int numToRound, const int multiple)
{
  if (multiple == 0) {

    return numToRound;
  }

  int remainder { numToRound % multiple };

  if (remainder == 0) {

    return numToRound;
  }

  return numToRound + multiple - remainder;
}

int findNearestDivisor(const int numToRound, const int divisor)
{

  if (numToRound > divisor) {

    return divisor;
  }

  int result = numToRound;

  while (divisor % result != 0) {

    ++result;
  }

  return result;
}

}

dim3 CAGPUUtils::Host::getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

dim3 CAGPUUtils::Host::getBlockSize(const int colsNum, const int rowsNum)
{
  const CAGPUDeviceProperties& deviceProperties = CAGPUContext::getInstance().getDeviceProperties();
  return getBlockSize(colsNum, rowsNum, deviceProperties.cudaCores / deviceProperties.maxBlocksPerSM);
}

dim3 CAGPUUtils::Host::getBlockSize(const int colsNum, const int rowsNum, const int maxThreadsPerBlock)
{
  const CAGPUDeviceProperties& deviceProperties = CAGPUContext::getInstance().getDeviceProperties();
  int xThreads = min(colsNum, deviceProperties.maxThreadsDim.x);
  int yThreads = min(rowsNum, deviceProperties.maxThreadsDim.y);
  const int totalThreads = roundUp(min(xThreads * yThreads, maxThreadsPerBlock),
      deviceProperties.warpSize);

  if (xThreads > yThreads) {

    xThreads = findNearestDivisor(xThreads, totalThreads);
    yThreads = totalThreads / xThreads;

  } else {

    yThreads = findNearestDivisor(yThreads, totalThreads);
    xThreads = totalThreads / yThreads;
  }

  return dim3 { static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads) };
}

dim3 CAGPUUtils::Host::getBlocksGrid(const dim3 &threadsPerBlock, const int rowsNum)
{

  return getBlocksGrid(threadsPerBlock, rowsNum, 1);
}

dim3 CAGPUUtils::Host::getBlocksGrid(const dim3 &threadsPerBlock, const int rowsNum, const int colsNum)
{

  return dim3 { 1 + (rowsNum - 1) / threadsPerBlock.x, 1 + (colsNum - 1) / threadsPerBlock.y };
}

void CAGPUUtils::Host::gpuMalloc(void **p, const int size)
{
  checkCUDAError(cudaMalloc(p, size), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuFree(void *p)
{
  checkCUDAError(cudaFree(p), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuMemset(void *p, int value, int size)
{
  checkCUDAError(cudaMemset(p, value, size), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuMemcpyHostToDevice(void *dst, const void *src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuMemcpyHostToDeviceAsync(void *dst, const void *src, int size, CAGPUStream &stream)
{
  checkCUDAError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream.get()), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuMemcpyDeviceToHost(void *dst, const void *src, int size)
{
  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuStartProfiler()
{
  checkCUDAError(cudaProfilerStart(), __FILE__, __LINE__);
}

void CAGPUUtils::Host::gpuStopProfiler()
{
  checkCUDAError(cudaProfilerStop(), __FILE__, __LINE__);
}

GPU_DEVICE int CAGPUUtils::Device::getLaneIndex()
{
  uint32_t laneIndex;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneIndex));
  return static_cast<int>(laneIndex);
}

GPU_DEVICE int CAGPUUtils::Device::shareToWarp(const int value, const int laneIndex)
{
  return __shfl(value, laneIndex);
}

GPU_DEVICE int CAGPUUtils::Device::gpuAtomicAdd(int *p, const int incrementSize)
{
  return atomicAdd(p, incrementSize);
}
