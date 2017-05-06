/// \file CACUDAArray.h
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

#ifndef TRAKINGITSU_INCLUDE_CUDA_CACUDAARRAY_H_
#define TRAKINGITSU_INCLUDE_CUDA_CACUDAARRAY_H_

#include <sstream>
#include <cuda_runtime.h>

template<typename T>
class CACUDAArray
  final
  {
    public:
      CACUDAArray();
      CACUDAArray(const int);
      CACUDAArray(const T* const, const int);
      ~CACUDAArray();

      __host__ __device__ const T* const get() const;
      __host__ __device__ T operator[](const int) const;

    private:
      T *mArrayPointer;
      int *mSize;
      int mCapacity;
  };

  template<typename T>
  CACUDAArray<T>::CACUDAArray()
      : mArrayPointer { nullptr }, mSize{ nullptr }, mCapacity { 0 }
  {
    // Nothing to do
  }

  template<typename T>
  CACUDAArray<T>::CACUDAArray(const int capacity)
      : CACUDAArray{ nullptr, capacity }
  {
    // Nothing to do
  }

  template<typename T>
  CACUDAArray<T>::CACUDAArray(const T* const source, const int size) :
    mCapacity{ size }
  {
    cudaError_t error;

    error = cudaMalloc((void **) &mArrayPointer, size * sizeof(T));

    if (error != cudaSuccess) {

      std::ostringstream errorString;

      errorString << "cudaMalloc returned error " << cudaGetErrorString(error) << " (code " << error << "), line("
          << __LINE__ << ")" << std::endl;

      throw std::runtime_error(errorString.str());
    }

    error = cudaMalloc((void **) &mSize, sizeof(int));

    if (error != cudaSuccess) {

      cudaFree(mArrayPointer);

      std::ostringstream errorString;

      errorString << "cudaMalloc returned error " << cudaGetErrorString(error) << " (code " << error << "), line("
          << __LINE__ << ")" << std::endl;

      throw std::runtime_error(errorString.str());
    }

    if(source != nullptr) {

      error = cudaMemcpy(mArrayPointer, source, size, cudaMemcpyHostToDevice);

      if (error != cudaSuccess) {

        cudaFree(mArrayPointer);
        cudaFree(mSize);

        std::ostringstream errorString;

        errorString << "cudaMemcpy returned error " << cudaGetErrorString(error) << " (code " << error << "), line("
            << __LINE__ << ")" << std::endl;

        throw std::runtime_error(errorString.str());
      }

      error = cudaMemcpy(mSize, &size, sizeof(int), cudaMemcpyHostToDevice);

      if (error != cudaSuccess) {

        cudaFree(mArrayPointer);
        cudaFree(mSize);

        std::ostringstream errorString;

        errorString << "cudaMemcpy returned error " << cudaGetErrorString(error) << " (code " << error << "), line("
            << __LINE__ << ")" << std::endl;

        throw std::runtime_error(errorString.str());
      }

    } else {

      error = cudaMemset(mSize, 0, sizeof(int));

      if (error != cudaSuccess) {

        cudaFree(mArrayPointer);
        cudaFree(mSize);

        std::ostringstream errorString;

        errorString << "cudaMemset returned error " << cudaGetErrorString(error) << " (code " << error << "), line("
            << __LINE__ << ")" << std::endl;

        throw std::runtime_error(errorString.str());
      }
    }
  }

  template<typename T>
  CACUDAArray<T>::~CACUDAArray()
  {
    if (mArrayPointer != nullptr) {

      cudaFree(mArrayPointer);
    }

    if(mSize != nullptr) {

      cudaFree(mSize);
    }
  }

  template<typename T>
  __host__ __device__ inline const T* const CACUDAArray<T>::get() const
  {

    return mArrayPointer;
  }

  template<typename T>
  __host__ __device__ inline T CACUDAArray<T>::operator[](const int index) const
  {

    return mArrayPointer[index];
  }

#endif /* TRAKINGITSU_INCLUDE_CUDA_CACUDAARRAY_H_ */
