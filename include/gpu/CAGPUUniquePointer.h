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

namespace {
template<typename T>
struct CAGPUUniquePointerTraits final
{
    typedef T* InternalPointer;

    GPU_HOST_DEVICE static constexpr T&
    getReference(const InternalPointer& internalPointer) noexcept
    { return const_cast<T&>(*internalPointer); }

    GPU_HOST_DEVICE static constexpr T*
    getPointer(const InternalPointer& internalPointer) noexcept
    { return const_cast<T*>(internalPointer); }
};
}

template<typename T>
class CAGPUUniquePointer final
{
  typedef CAGPUUniquePointerTraits<T> PointerTraits;

  public:
    CAGPUUniquePointer();
    explicit CAGPUUniquePointer(const T&);
    ~CAGPUUniquePointer();

    CAGPUUniquePointer(const CAGPUUniquePointer&) = delete;
    CAGPUUniquePointer &operator=(const CAGPUUniquePointer&) = delete;

    CAGPUUniquePointer(CAGPUUniquePointer&&);
    CAGPUUniquePointer &operator=(CAGPUUniquePointer&&);

    GPU_HOST_DEVICE T* get() noexcept;
    GPU_HOST_DEVICE const T* get() const noexcept;
    GPU_HOST_DEVICE T& operator*() noexcept;
    GPU_HOST_DEVICE const T& operator*() const noexcept;

  protected:
    void destroy();

  private:
    typename PointerTraits::InternalPointer mDevicePointer;

};

template<typename T>
CAGPUUniquePointer<T>::CAGPUUniquePointer()
    : mDevicePointer { nullptr }
{
  // Nothing to do
}

template<typename T>
CAGPUUniquePointer<T>::CAGPUUniquePointer(const T &ref)
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
GPU_HOST_DEVICE T* CAGPUUniquePointer<T>::get() noexcept
{
  return PointerTraits::getPointer(mDevicePointer);
}

template<typename T>
GPU_HOST_DEVICE const T* CAGPUUniquePointer<T>::get() const noexcept
{
  return PointerTraits::getPointer(mDevicePointer);
}

template<typename T>
GPU_HOST_DEVICE T& CAGPUUniquePointer<T>::operator*() noexcept
{
  return PointerTraits::getReference(mDevicePointer);
}

template<typename T>
GPU_HOST_DEVICE const T& CAGPUUniquePointer<T>::operator*() const noexcept
{
  return PointerTraits::getReference(mDevicePointer);
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_ */
