/// \file CAGPUUniquePointer.h
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

#ifndef TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_
#define TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_

#include "CAGPUUtils.h"

template<typename T>
class CAGPUUniquePointer
{
  public:
    CAGPUUniquePointer();
    explicit CAGPUUniquePointer(T&);
    ~CAGPUUniquePointer();

    CAGPUUniquePointer(const CAGPUUniquePointer&) = delete;
    CAGPUUniquePointer &operator=(const CAGPUUniquePointer&) = delete;

    CAGPUUniquePointer(CAGPUUniquePointer&&);
    CAGPUUniquePointer &operator=(CAGPUUniquePointer&&);

    T* get();
    T& operator *();

  protected:
    void destroy();

  private:
    T *mDevicePointer;

};

template<typename T>
CAGPUUniquePointer<T>::CAGPUUniquePointer()
    : mDevicePointer { nullptr }
{
  // Nothing to do
}

template<typename T>
CAGPUUniquePointer<T>::CAGPUUniquePointer(T &ref)
{
  try {

    CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void**>(&mDevicePointer), sizeof(T));
    CAGPUUtils::Host::gpuMemcpyHostToDevice(mDevicePointer, &ref, sizeof(T));

  } catch (...) {

    destroy();

    throw;
  }
}

template<typename T>
CAGPUUniquePointer<T>::~CAGPUUniquePointer()
{
  destroy();
}

template<typename T>
CAGPUUniquePointer<T>::CAGPUUniquePointer(CAGPUUniquePointer<T>&& other)
    : mDevicePointer { other.mDevicePointer }
{
  // Nothing to do
}

template<typename T>
CAGPUUniquePointer<T>& CAGPUUniquePointer<T>::operator =(CAGPUUniquePointer<T>&& other)
{
  mDevicePointer = other.mDevicePointer;
  other.mDevicePointer = nullptr;

  return *this;
}

template<typename T>
void CAGPUUniquePointer<T>::destroy()
{
  if (mDevicePointer != nullptr) {

    CAGPUUtils::Host::gpuFree(mDevicePointer);
  }
}

template<typename T>
T *CAGPUUniquePointer<T>::get()
{
  return mDevicePointer;
}

template<typename T>
T &CAGPUUniquePointer<T>::operator *()
{
  return *mDevicePointer;
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_ */
