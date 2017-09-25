/// \file Utils.h
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

#ifndef TRACKINGITSU_INCLUDE_GPU_UTILS_H_
#define TRACKINGITSU_INCLUDE_GPU_UTILS_H_

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/Stream.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

namespace Utils {

namespace Host {
dim3 getBlockSize(const int);
dim3 getBlockSize(const int, const int);
dim3 getBlockSize(const int, const int, const int);
dim3 getBlocksGrid(const dim3&, const int);
dim3 getBlocksGrid(const dim3&, const int, const int);

void gpuMalloc(void**, const int);
void gpuFree(void*);
void gpuMemset(void *, int, int);
void gpuMemcpyHostToDevice(void *, const void *, int);
void gpuMemcpyHostToDeviceAsync(void *, const void *, int, Stream&);
void gpuMemcpyDeviceToHost(void *, const void *, int);
void gpuStartProfiler();
void gpuStopProfiler();
}

namespace Device {
GPU_DEVICE int getLaneIndex();
GPU_DEVICE int shareToWarp(const int, const int);
GPU_DEVICE int gpuAtomicAdd(int*, const int);
}
}

}
}
}
}

#endif /* TRACKINGITSU_INCLUDE_GPU_UTILS_H_ */
