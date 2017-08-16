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

#include <assert.h>
#include <new>
#include <type_traits>
#include <vector>

#include "CADefinitions.h"
#include "CAGPUStream.h"
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
      CAGPUVector(const T* const, const int, const int = 0);GPU_HOST_DEVICE ~CAGPUVector();

      CAGPUVector(const CAGPUVector&) = delete;
      CAGPUVector &operator=(const CAGPUVector&) = delete;

      GPU_HOST_DEVICE CAGPUVector(CAGPUVector&&);
      CAGPUVector &operator=(CAGPUVector&&);

      int getSizeFromDevice() const;
      void resize(const int);
      void copyIntoVector(std::vector<T>&, const int);

      GPU_HOST_DEVICE T* get() const;
      GPU_HOST_DEVICE int capacity() const;
      GPU_HOST_DEVICE CAGPUVector<T> getWeakCopy() const;
      GPU_DEVICE T& operator[](const int) const;
      GPU_DEVICE int size() const;
      GPU_DEVICE int extend(const int) const;

      template<typename ...Args>
      GPU_DEVICE void emplace(const int, Args&&...);

    protected:
      void destroy();

    private:
      GPU_HOST_DEVICE CAGPUVector(const CAGPUVector&, const bool);

      T *mArrayPointer;
      int *mDeviceSize;
      int mCapacity;
      bool mIsWeak;
  };

  template<typename T>
  CAGPUVector<T>::CAGPUVector()
      : mArrayPointer { nullptr }, mDeviceSize { nullptr }, mCapacity { 0 }, mIsWeak { true }
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
      : mCapacity { size }, mIsWeak { false }
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
  CAGPUVector<T>::CAGPUVector(const CAGPUVector& other, const bool isWeak)
      : mArrayPointer { other.mArrayPointer }, mDeviceSize { other.mDeviceSize }, mCapacity { other.mCapacity }, mIsWeak {
          isWeak }
  {
    //Nothing to do
  }

  template<typename T>
  GPU_HOST_DEVICE CAGPUVector<T>::~CAGPUVector()
  {
    if (mIsWeak) {

      return;

    } else {
#if defined(TRACKINGITSU_GPU_DEVICE)
      assert(0);
#else
      destroy();
#endif
    }
  }

  template<typename T>
  GPU_HOST_DEVICE CAGPUVector<T>::CAGPUVector(CAGPUVector<T> &&other)
      : mArrayPointer { other.mArrayPointer }, mDeviceSize { other.mDeviceSize }, mCapacity { other.mCapacity }, mIsWeak {
          other.mIsWeak }
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
    mIsWeak = other.mIsWeak;

    other.mArrayPointer = nullptr;
    other.mDeviceSize = nullptr;

    return *this;
  }

  template<typename T>
  int CAGPUVector<T>::getSizeFromDevice() const
  {
    int size;
    CAGPUUtils::Host::gpuMemcpyDeviceToHost(&size, mDeviceSize, sizeof(int));

    return size;
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
  GPU_HOST_DEVICE inline CAGPUVector<T> CAGPUVector<T>::getWeakCopy() const
  {
    return CAGPUVector { *this, true };
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
    assert(size() <= mCapacity);

    return startIndex;
  }

  template<typename T>
  template<typename ...Args>
  GPU_DEVICE void CAGPUVector<T>::emplace(const int index, Args&&... arguments)
  {

    new (mArrayPointer + index) T(std::forward < Args > (arguments)...);
  }

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_ */
