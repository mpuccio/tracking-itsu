/// \file CAGPUVector.h
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

#ifndef TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_
#define TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_

#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#include "CADefinitions.h"
#include "CAGPUUtils.h"

template<typename T>
class CAGPUVector
    final
    {
      static_assert(std::is_trivially_destructible<T>::value,
          "CAGPUVector only supports trivially destructible objects.");

    public:
      CAGPUVector();
      explicit CAGPUVector(const int, const int = 0);
      CAGPUVector(const T* const, const int, const int = 0);
      ~CAGPUVector();

      CAGPUVector(const CAGPUVector&) = delete;
      CAGPUVector &operator=(const CAGPUVector&) = delete;

      CAGPUVector(CAGPUVector&&);
      CAGPUVector &operator=(CAGPUVector&&);

      std::unique_ptr<int, void (*)(void*)> getSizeFromDevice() const;
      void resize(const int);
      void copyIntoVector(std::vector<T>&, const int);

      GPU_HOST_DEVICE T* get() const;
      GPU_HOST_DEVICE int capacity() const;
      GPU_DEVICE T& operator[](const int) const;
      GPU_DEVICE int size() const;
      GPU_DEVICE int extend(const int) const;
      template<typename ...Args>
      GPU_DEVICE void emplace(const int, Args&&...);

    protected:
      void destroy();

    private:
      T *mArrayPointer;
      int *mDeviceSize;
      int mCapacity;
  };

  template<typename T>
  CAGPUVector<T>::CAGPUVector()
      : mArrayPointer { nullptr }, mDeviceSize { nullptr }, mCapacity { 0 }
  {
    // Nothing to do
  }

  template<typename T>
  CAGPUVector<T>::CAGPUVector(const int capacity, const int initialSize)
      : CAGPUVector { nullptr, capacity, initialSize }
  {
    // Nothing to do
  }

  template<typename T>
  CAGPUVector<T>::CAGPUVector(const T* const source, const int size, const int initialSize)
      : mCapacity { size }
  {
    try {

      CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void **>(&mArrayPointer), size * sizeof(T));
      CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void **>(&mDeviceSize), sizeof(int));

      if (source != nullptr) {

        CAGPUUtils::Host::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
        CAGPUUtils::Host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));

      } else {

        CAGPUUtils::Host::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(int));

      }

    } catch (...) {

      destroy();

      throw;
    }
  }

  template<typename T>
  CAGPUVector<T>::~CAGPUVector()
  {
    destroy();
  }

  template<typename T>
  CAGPUVector<T>::CAGPUVector(CAGPUVector<T> &&other)
      : mArrayPointer { other.mArrayPointer }, mDeviceSize { other.mDeviceSize }, mCapacity { other.mCapacity }
  {
    other.mArrayPointer = nullptr;
    other.mDeviceSize = nullptr;
  }

  template<typename T>
  CAGPUVector<T> &CAGPUVector<T>::operator=(CAGPUVector<T> &&other)
  {
    destroy();

    mArrayPointer = other.mArrayPointer;
    mDeviceSize = other.mDeviceSize;
    mCapacity = other.mCapacity;

    other.mArrayPointer = nullptr;
    other.mDeviceSize = nullptr;

    return *this;
  }

  template<typename T>
  std::unique_ptr<int, void (*)(void*)> CAGPUVector<T>::getSizeFromDevice() const
  {
    int *primitiveHostSizePointer = nullptr;

    try {

      primitiveHostSizePointer = static_cast<int *>(malloc(sizeof(int)));
      CAGPUUtils::Host::gpuMemcpyDeviceToHost(primitiveHostSizePointer, mDeviceSize, sizeof(int));

      return std::unique_ptr<int, void (*)(void*)> {
        primitiveHostSizePointer, free };

    } catch(...) {

      free(primitiveHostSizePointer);

      throw;
    }
  }

  template<typename T>
  void CAGPUVector<T>::resize(const int size)
  {
    CAGPUUtils::Host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));
  }

  template<typename T>
  void CAGPUVector<T>::copyIntoVector(std::vector<T> &destinationArray, const int size)
  {

    T *hostPrimitivePointer = nullptr;

    try {

      hostPrimitivePointer = static_cast<T *>(malloc(size * sizeof(T)));
      CAGPUUtils::Host::gpuMemcpyDeviceToHost(hostPrimitivePointer, mArrayPointer, size * sizeof(T));

      destinationArray = std::move(std::vector<T>(hostPrimitivePointer, hostPrimitivePointer + size));

    } catch (...) {

      if (hostPrimitivePointer != nullptr) {

        free(hostPrimitivePointer);
      }

      throw;
    }
  }

  template<typename T>
  inline void CAGPUVector<T>::destroy()
  {
    if (mArrayPointer != nullptr) {

      CAGPUUtils::Host::gpuFree(mArrayPointer);
    }

    if (mDeviceSize != nullptr) {

      CAGPUUtils::Host::gpuFree(mDeviceSize);
    }
  }

  template<typename T>
  GPU_HOST_DEVICE inline T* CAGPUVector<T>::get() const
  {
    return mArrayPointer;
  }

  template<typename T>
  GPU_HOST_DEVICE inline int CAGPUVector<T>::capacity() const
  {
    return mCapacity;
  }

  template<typename T>
  GPU_DEVICE inline T& CAGPUVector<T>::operator[](const int index) const
  {
    return mArrayPointer[index];
  }

  template<typename T>
  GPU_DEVICE inline int CAGPUVector<T>::size() const
  {
    return *mDeviceSize;
  }

  template<typename T>
  GPU_DEVICE int CAGPUVector<T>::extend(const int sizeIncrement) const
  {
    const int startIndex = CAGPUUtils::Device::gpuAtomicAdd(mDeviceSize, sizeIncrement);

    if (size() > mCapacity) {

      return -1; //TODO: error handling

    } else {

      return startIndex;
    }
  }

  template<typename T>
  template<typename ...Args>
  GPU_DEVICE void CAGPUVector<T>::emplace(const int index, Args&&... arguments)
  {

    new (mArrayPointer + index) T(std::forward < Args > (arguments)...);
  }

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_ */
