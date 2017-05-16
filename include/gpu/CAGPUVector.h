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

#include <new>
#include <type_traits>

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
      CAGPUVector(const int);
      CAGPUVector(const T* const, const int);
      ~CAGPUVector();

      GPU_DEVICE T* get() const;
      GPU_DEVICE T operator[](const int) const;
      GPU_DEVICE int size() const;
      GPU_DEVICE int extend(const int) const;
      template<typename ...Args>
      GPU_DEVICE void insert(const int, Args&&...);

    protected:
      void destroy();

    private:
      T *mArrayPointer;
      int *mSize;
      int mCapacity;
  };

  template<typename T>
  CAGPUVector<T>::CAGPUVector()
      : mArrayPointer { nullptr }, mSize { nullptr }, mCapacity { 0 }
  {
    // Nothing to do
  }

  template<typename T>
  CAGPUVector<T>::CAGPUVector(const int capacity)
      : CAGPUVector { nullptr, capacity }
  {
    // Nothing to do
  }

  template<typename T>
  CAGPUVector<T>::CAGPUVector(const T* const source, const int size)
      : mCapacity { size }
  {
    try {

      CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void **>(&mArrayPointer), size * sizeof(T));
      CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void **>(&mSize), sizeof(int));

      if (source != nullptr) {

        CAGPUUtils::Host::gpuMemcpyHostToDevice(mArrayPointer, source, size);
        CAGPUUtils::Host::gpuMemcpyHostToDevice(mSize, &size, sizeof(int));

      } else {

        CAGPUUtils::Host::gpuMemset(mSize, 0, sizeof(int));
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
  GPU_DEVICE inline T* CAGPUVector<T>::get() const
  {
    return mArrayPointer;
  }

  template<typename T>
  GPU_DEVICE inline T CAGPUVector<T>::operator[](const int index) const
  {
    return mArrayPointer[index];
  }

  template<typename T>
  GPU_DEVICE inline int CAGPUVector<T>::size() const
  {
    return *mSize;
  }

  template<typename T>
  GPU_DEVICE int CAGPUVector<T>::extend(const int sizeIncrement) const
  {
    const int startIndex = CAGPUUtils::Device::gpuAtomicAdd(mSize, sizeIncrement * sizeof(T));

    if (size() > mCapacity) {

      return -1; //TODO: error handling

    } else {

      return startIndex;
    }
  }

  template<typename T>
  inline void CAGPUVector<T>::destroy()
  {

    if (mArrayPointer != nullptr) {

      CAGPUUtils::Host::gpuFree(mArrayPointer);
    }

    if (mSize != nullptr) {

      CAGPUUtils::Host::gpuFree(mSize);
    }
  }

  template<typename T>
  template<typename ...Args>
  GPU_DEVICE void CAGPUVector<T>::insert(const int index, Args&&... arguments)
  {

    new (mArrayPointer + index) T(std::forward < Args > (arguments)...);
  }

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_ */
