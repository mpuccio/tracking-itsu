/// \file CAGPUArray.h
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

#ifndef TRAKINGITSU_INCLUDE_GPU_CAGPUARRAY_H_
#define TRAKINGITSU_INCLUDE_GPU_CAGPUARRAY_H_

#include "CAGPUUtils.h"

template<typename T>
class CAGPUArray
  final
  {
    public:
      CAGPUArray();
      CAGPUArray(const int);
      CAGPUArray(const T* const, const int);
      ~CAGPUArray();

      const T* const get() const;
      T operator[](const int) const;

    protected:
      void destroy();

    private:
      T *mArrayPointer;
      int *mSize;
      int mCapacity;
  };

  template<typename T>
  CAGPUArray<T>::CAGPUArray()
      : mArrayPointer { nullptr }, mSize { nullptr }, mCapacity { 0 }
  {
    // Nothing to do
  }

  template<typename T>
  CAGPUArray<T>::CAGPUArray(const int capacity)
      : CAGPUArray { nullptr, capacity }
  {
    // Nothing to do
  }

  template<typename T>
  CAGPUArray<T>::CAGPUArray(const T* const source, const int size)
      : mCapacity { size }
  {
    try {

      CAGPUUtils::gpuMalloc((void **) &mArrayPointer, size * sizeof(T));
      CAGPUUtils::gpuMalloc((void **) &mSize, sizeof(int));

      if (source != nullptr) {

        CAGPUUtils::gpuMemcpyHostToDevice(mArrayPointer, source, size);
        CAGPUUtils::gpuMemcpyHostToDevice(mSize, &size, sizeof(int));

      } else {

        CAGPUUtils::gpuMemset(mSize, 0, sizeof(int));
      }

    } catch (...) {

      destroy();

      throw;
    }
  }

  template<typename T>
  CAGPUArray<T>::~CAGPUArray()
  {
    destroy();
  }

  template<typename T>
  inline const T* const CAGPUArray<T>::get() const
  {
    return mArrayPointer;
  }

  template<typename T>
  inline T CAGPUArray<T>::operator[](const int index) const
  {
    return mArrayPointer[index];
  }

  template<typename T>
  inline void CAGPUArray<T>::destroy() {

    if (mArrayPointer != nullptr) {

      CAGPUUtils::gpuFree(mArrayPointer);
    }

    if (mSize != nullptr) {

      CAGPUUtils::gpuFree(mSize);
    }
  }

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUARRAY_H_ */
