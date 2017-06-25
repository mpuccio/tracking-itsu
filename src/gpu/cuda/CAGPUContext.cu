/// \file CAGPUContext.cu
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

#include "CAGPUContext.h"

#include <sstream>

#include <cuda_runtime.h>

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
}

CAGPUContext::CAGPUContext()
{
  checkCUDAError(cudaGetDeviceCount(&mDevicesNum), __FILE__, __LINE__);

  if (mDevicesNum == 0) {

    throw std::runtime_error { "There are no available device(s) that support CUDA\n" };
  }

  mDeviceProperties.resize(mDevicesNum, CAGPUDeviceProperties { });

  int currentDeviceIndex;
  checkCUDAError(cudaGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  for (int iDevice { 0 }; iDevice < mDevicesNum; ++iDevice) {

    cudaDeviceProp deviceProperties;

    checkCUDAError(cudaSetDevice(iDevice), __FILE__, __LINE__);
    checkCUDAError(cudaGetDeviceProperties(&deviceProperties, iDevice), __FILE__, __LINE__);

    mDeviceProperties[iDevice].name = deviceProperties.name;
    mDeviceProperties[iDevice].gpuProcessors = deviceProperties.multiProcessorCount;
    mDeviceProperties[iDevice].globalMemorySize = deviceProperties.totalGlobalMem;
    mDeviceProperties[iDevice].constantMemorySize = deviceProperties.totalConstMem;
    mDeviceProperties[iDevice].sharedMemorySize = deviceProperties.sharedMemPerBlock;
    mDeviceProperties[iDevice].maxClockRate = deviceProperties.memoryClockRate;
    mDeviceProperties[iDevice].busWidth = deviceProperties.memoryBusWidth;
    mDeviceProperties[iDevice].l2CacheSize = deviceProperties.l2CacheSize;
    mDeviceProperties[iDevice].registersPerBlock = deviceProperties.regsPerBlock;
    mDeviceProperties[iDevice].warpSize = deviceProperties.warpSize;
    mDeviceProperties[iDevice].maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    mDeviceProperties[iDevice].maxThreadsDim = dim3 { static_cast<unsigned int>(deviceProperties.maxThreadsDim[0]),
        static_cast<unsigned int>(deviceProperties.maxThreadsDim[1]),
        static_cast<unsigned int>(deviceProperties.maxThreadsDim[2]) };
    mDeviceProperties[iDevice].maxGridDim = dim3 { static_cast<unsigned int>(deviceProperties.maxGridSize[0]),
        static_cast<unsigned int>(deviceProperties.maxGridSize[1]),
        static_cast<unsigned int>(deviceProperties.maxGridSize[2]) };
  }

  checkCUDAError(cudaSetDevice(currentDeviceIndex), __FILE__, __LINE__);
}

CAGPUContext& CAGPUContext::getInstance()
{
  static CAGPUContext gpuContext;
  return gpuContext;
}

const CAGPUDeviceProperties& CAGPUContext::getDeviceProperties()
{
  int currentDeviceIndex;
  checkCUDAError(cudaGetDevice(&currentDeviceIndex), __FILE__, __LINE__);

  return getDeviceProperties(currentDeviceIndex);
}

const CAGPUDeviceProperties& CAGPUContext::getDeviceProperties(const int deviceIndex)
{
  return mDeviceProperties[deviceIndex];
}
