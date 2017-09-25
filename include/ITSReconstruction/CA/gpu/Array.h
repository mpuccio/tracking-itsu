/// \file Array.h
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

#ifndef TRAKINGITSU_INCLUDE_GPU_ARRAY_H_
#define TRAKINGITSU_INCLUDE_GPU_ARRAY_H_

#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

namespace {
template<typename T, std::size_t Size>
struct ArrayTraits final
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
struct Array
    final
    {
      typedef ArrayTraits<T, Size> Trait;

      GPU_HOST_DEVICE T* data() noexcept;
      GPU_HOST_DEVICE const T* data() const noexcept;
      GPU_HOST_DEVICE T& operator[](const int) noexcept;
      GPU_HOST_DEVICE constexpr T& operator[](const int) const noexcept;
      GPU_HOST_DEVICE std::size_t size() const noexcept;

      typename Trait::InternalArray arrayPointer;
  };

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE T* Array<T, Size>::data() noexcept
  {
    return Trait::getPointer(arrayPointer);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE const T* Array<T, Size>::data() const noexcept
  {
    return Trait::getPointer(arrayPointer);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE constexpr T& Array<T, Size>::operator[](const int index) const noexcept
  {
    return Trait::getReference(arrayPointer, index);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE T& Array<T, Size>::operator[](const int index) noexcept
  {
    return Trait::getReference(arrayPointer, index);
  }

  template<typename T, std::size_t Size>
  GPU_HOST_DEVICE std::size_t Array<T, Size>::size() const noexcept
  {
    return Size;
  }

}
}
}
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_ */
