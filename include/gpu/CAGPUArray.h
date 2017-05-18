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

#include "CADefinitions.h"

namespace {
template<typename T, std::size_t Size>
struct CAGPUArrayTraits
{
    typedef T InternalArray[Size];

    GPU_HOST_DEVICE static constexpr T&
    getReference(const InternalArray& internalArray, std::size_t index) noexcept
    { return const_cast<T&>(internalArray[index]); }

    GPU_HOST_DEVICE static constexpr T*
    getPointer(const InternalArray& internalArray) noexcept
    { return const_cast<T*>(internalArray); }
};
}

template<typename T, std::size_t Size>
struct CAGPUArray
    final
    {
      typedef CAGPUArrayTraits<T, Size> ArrayTraits;

      GPU_HOST_DEVICE T* data() noexcept;
      GPU_HOST_DEVICE const T* data() const noexcept;
      GPU_HOST_DEVICE T& operator[](const int) noexcept;
      GPU_HOST_DEVICE constexpr T& operator[](const int) const noexcept;
      GPU_HOST_DEVICE std::size_t size() const noexcept;

      typename ArrayTraits::InternalArray arrayPointer;
  };

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE T* CAGPUArray<T, Size>::data() noexcept
  {
    return ArrayTraits::getPointer(arrayPointer);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE const T* CAGPUArray<T, Size>::data() const noexcept
  {
    return ArrayTraits::getPointer(arrayPointer);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE constexpr T& CAGPUArray<T, Size>::operator[](const int index) const noexcept
  {
    return ArrayTraits::getReference(arrayPointer, index);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE T& CAGPUArray<T, Size>::operator[](const int index) noexcept
  {
    return ArrayTraits::getReference(arrayPointer, index);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE std::size_t CAGPUArray<T, Size>::size() const noexcept
  {
    return Size;
  }

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_ */
