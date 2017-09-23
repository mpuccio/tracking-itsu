/// \file Context.h
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

#ifndef TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_
#define TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_

#include <string>
#include <vector>

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

struct DeviceProperties final
{
    std::string name;
    int gpuProcessors;
    int cudaCores;
    long globalMemorySize;
    long constantMemorySize;
    long sharedMemorySize;
    long maxClockRate;
    int busWidth;
    long l2CacheSize;
    long registersPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxBlocksPerSM;
    dim3 maxThreadsDim;
    dim3 maxGridDim;
};

class Context final
{
  public:
    static Context& getInstance();

    Context(const Context&);
    Context& operator=(const Context&);

    const DeviceProperties& getDeviceProperties();
    const DeviceProperties& getDeviceProperties(const int);

  private:
    Context();
    ~Context() = default;

    int mDevicesNum;
    std::vector<DeviceProperties> mDeviceProperties;
};

}
}
}
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_ */
