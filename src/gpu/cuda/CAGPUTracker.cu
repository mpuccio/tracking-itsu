/// \file CAGPUTracker.cu
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

#include "CATracker.h"

#include <array>
#include <sstream>
#include <iostream>

#include <cuda_runtime.h>

#include "CAConstants.h"
#include "CAGPUPrimaryVertexContext.h"
#include "CAGPUVector.h"
#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"
#include "CAPrimaryVertexContext.h"
#include "CATrackingUtils.h"

//TODO: this must be refined with runtime device queries or with careful planning
constexpr int WarpSize { 32 };
constexpr int MaxXThreads { 128 };
constexpr int MaxYThreads { 128 };
constexpr int MaxThreadsPerBlock { 128 };

__host__ __device__ dim3 getBlockSize(const int colsNum, const int rowsNum)
{
  int xThreads = min(colsNum, MaxXThreads);
  int yThreads = min(rowsNum, MaxYThreads);
  const int totalThreads = min(CAMathUtils::roundUp(xThreads * yThreads, WarpSize), MaxThreadsPerBlock);

  if (xThreads > yThreads) {

    xThreads = CAMathUtils::findNearestDivisor(xThreads, totalThreads);
    yThreads = totalThreads / xThreads;

  } else {

    yThreads = CAMathUtils::findNearestDivisor(yThreads, totalThreads);
    xThreads = totalThreads / yThreads;
  }

  return dim3 { static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads) };
}

__host__ __device__ dim3 getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

__device__ int getLaneIndex()
{
  return (threadIdx.x + threadIdx.y * blockDim.x) % WarpSize;
}

__device__ int shareToWarp(int value, int leaderIndex)
{
  return __shfl(value, leaderIndex);
}

__global__ void layerTrackletsKernel(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    float3 primaryVertex, bool dryRun)
{
  int currentClusterIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);

  if (currentClusterIndex < primaryVertexContext.getClusters()[layerIndex].size()) {

    const CACluster& currentCluster { primaryVertexContext.getClusters()[layerIndex][currentClusterIndex] };

    /*if (mUsedClustersTable[currentCluster.clusterId] != CAConstants::ITS::UnusedIndex) {

     continue;
     }*/

    const float tanLambda { (currentCluster.zCoordinate - primaryVertex.z) / currentCluster.rCoordinate };
    const float directionZIntersection { tanLambda
        * ((CAConstants::ITS::LayersRCoordinate())[layerIndex + 1] - currentCluster.rCoordinate)
        + currentCluster.zCoordinate };

    const GPU_ARRAY<int, 4> selectedBinsRect { CATrackingUtils::getBinsRect(currentCluster, layerIndex,
        directionZIntersection) };

    if (selectedBinsRect[0] != 0 || selectedBinsRect[1] != 0 || selectedBinsRect[2] != 0 || selectedBinsRect[3] != 0) {

      const int nextLayerClustersNum{ static_cast<int> (primaryVertexContext.getClusters()[layerIndex + 1].size()) };
      int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

      if (phiBinsNum < 0) {

        phiBinsNum += CAConstants::IndexTable::PhiBins;
      }

      int startIndex;
      int currentIndex = 0;

      if(!dryRun) {

        const int currentClusterTracklets = primaryVertexContext.getTrackletsPerClusterTable()[layerIndex][currentClusterIndex];

        if(currentClusterTracklets == 0) {

          return;
        }

        startIndex = primaryVertexContext.getTracklets()[layerIndex].extend(currentClusterTracklets);

        if(layerIndex > 0) {

          primaryVertexContext.getTrackletsLookupTable()[layerIndex - 1][currentClusterIndex] = startIndex;
        }
      }

      for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
          iPhiBin = ++iPhiBin == CAConstants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex { CAIndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };
        const int maxBinIndex { firstBinIndex + selectedBinsRect[2] - selectedBinsRect[0] + 1 };
        const int firstRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][firstBinIndex];
        const int maxRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][maxBinIndex];

        for(int iNextLayerCluster { firstRowClusterIndex }; iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

          const CACluster& nextCluster { primaryVertexContext.getClusters()[layerIndex + 1][iNextLayerCluster] };

          if (CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

            if(dryRun) {

              ++primaryVertexContext.getTrackletsPerClusterTable()[layerIndex][currentClusterIndex];

            } else {

              primaryVertexContext.getTracklets()[layerIndex].insert(startIndex + currentIndex,
                  currentClusterIndex, iNextLayerCluster, currentCluster, nextCluster);
              ++currentIndex;
            }
          }
        }
      }
    }
  }
}

template<>
class CATrackerTraits<true>
{
  private:
    typedef CAPrimaryVertexContext<true> Context;
  public:
    CATrackerTraits();

    void computeLayerTracklets(Context& primaryVertexContext, const int layerIndex,
        const std::array<float, 3> &primaryVertex);
    void postProcessTracklets(Context&);

  protected:
    ~CATrackerTraits();
  private:
    CAGPUArray<cudaStream_t, CAConstants::ITS::LayersNumber> streams;
};

CATrackerTraits<true>::CATrackerTraits()
{
  for (int iStream = 0; iStream < CAConstants::ITS::LayersNumber; ++iStream) {

    cudaStreamCreate(&streams[iStream]);
  }
}

CATrackerTraits<true>::~CATrackerTraits()
{
  for (int iStream = 0; iStream < CAConstants::ITS::LayersNumber; ++iStream) {

    cudaStreamDestroy(streams[iStream]);
  }
}

void CATrackerTraits<true>::computeLayerTracklets(Context& primaryVertexContext, const int layerIndex,
    const std::array<float, 3> &primaryVertex)
{
  const int clustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex].size()) };
  dim3 threadsPerBlock { getBlockSize(clustersNum) };
  dim3 blocksGrid { 1 + clustersNum / threadsPerBlock.x };

  cudaStream_t currentStream;
  cudaStreamCreate(&currentStream);

  layerTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, currentStream >>>(primaryVertexContext.getDeviceContext(),
      layerIndex, {primaryVertex[0], primaryVertex[1], primaryVertex[2]}, true);
  layerTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, currentStream >>>(primaryVertexContext.getDeviceContext(),
        layerIndex, {primaryVertex[0], primaryVertex[1], primaryVertex[2]}, false);

  cudaError_t error = cudaGetLastError();

  cudaStreamDestroy(currentStream);

  if (error != cudaSuccess) {

    std::ostringstream errorString { };
    errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;

    throw std::runtime_error { errorString.str() };
  }
}

void CATrackerTraits<true>::postProcessTracklets(Context& primaryVertexContext)
{
  cudaDeviceSynchronize();

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    std::unique_ptr<int, void (*)(void*)> sizeUniquePointer =
        primaryVertexContext.getDeviceTracklets()[iLayer].getSizeFromDevice();
    primaryVertexContext.getDeviceTracklets()[iLayer].copyIntoVector(primaryVertexContext.getTracklets()[iLayer],
        *sizeUniquePointer);

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      primaryVertexContext.getDeviceTrackletsLookupTable()[iLayer].copyIntoVector(
          primaryVertexContext.getTrackletsLookupTable()[iLayer], primaryVertexContext.getClusters()[iLayer + 1].size());
    }
  }
}
