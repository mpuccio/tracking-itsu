/// \file CAGPUTrackingAPI.cu
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

#include "CAGPUTrackingAPI.h"

#include <array>
#include <sstream>
#include <iostream>

#include <cuda_runtime.h>

#include "CAConstants.h"
#include "CAGPUVector.h"
#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"
#include "CATrackingUtils.h"

namespace {
//TODO: this must be refined with runtime device queries or with careful planning
constexpr int WarpSize { 32 };
constexpr int MaxXThreads { 128 };
constexpr int MaxYThreads { 128 };
constexpr int MaxThreadsPerBlock { 128 };

cudaStream_t streams[CAConstants::ITS::TrackletsPerRoad];

dim3 getBlockSize(const int colsNum, const int rowsNum)
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

__device__ int getLaneIndex()
{
  return (threadIdx.x + threadIdx.y * blockDim.x) % WarpSize;
}

__device__ int shareToWarp(int value, int leaderIndex)
{
  return __shfl(value, leaderIndex);
}

__global__ void trackletsKernel(CAGPUPrimaryVertexContext *primaryVertexContext, const int layerIndex,
    const int currentClusterIndex, const float tanLambda, const float directionZIntersection, const int minZBinIndex,
    const int minPhiBinIndex, const int maxZBinIndex, const int phiBinsNum)
{
  int currentXIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int currentYIndex = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);

  __shared__ int rowFirstBinClusterIndexes[CAConstants::IndexTable::PhiBins];
  __shared__ int rowMaxBinClusterIndexes[CAConstants::IndexTable::PhiBins];

  if (currentYIndex < phiBinsNum) {

    const int phiBinIndex { (minPhiBinIndex + currentYIndex) % CAConstants::IndexTable::PhiBins };
    rowFirstBinClusterIndexes[phiBinIndex] = primaryVertexContext->indexTables[layerIndex][CAIndexTableUtils::getBinIndex(minZBinIndex,
        phiBinIndex)];
    rowMaxBinClusterIndexes[phiBinIndex] = primaryVertexContext->indexTables[layerIndex][CAIndexTableUtils::getBinIndex(maxZBinIndex + 1,
        phiBinIndex)];

    __syncthreads();

    const int nextClusterIndex { rowFirstBinClusterIndexes[phiBinIndex] + currentXIndex };

    if (nextClusterIndex <= rowMaxBinClusterIndexes[phiBinIndex]) {

      const CACluster& currentCluster { primaryVertexContext->clusters[layerIndex][currentClusterIndex] };
      const CACluster& nextCluster { primaryVertexContext->clusters[layerIndex + 1][nextClusterIndex] };

      if (CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

        const int laneIndex { getLaneIndex() };
        const unsigned int mask { __ballot(1) };
        const int leaderIndex { __ffs(mask) - 1 };
        int startIndex { };

        if (laneIndex == leaderIndex) {

          startIndex = primaryVertexContext->tracklets[layerIndex].extend(__popc(mask));

          if (layerIndex > 0) {

            atomicMin(&primaryVertexContext->trackletsLookupTable[layerIndex - 1][currentClusterIndex], startIndex);
          }
        }

        startIndex = shareToWarp(startIndex, leaderIndex);

        primaryVertexContext->tracklets[layerIndex].insert(startIndex + __popc(mask & ((1 << laneIndex) - 1)), currentClusterIndex, nextClusterIndex,
            currentCluster, nextCluster);
      }
    }
  }
}
}

void CAGPUTrackingAPI::getTrackletsFromCluster(CAPrimaryVertexContext<true>& primaryVertexContext,
    const int currentLayerIndex, const int currentClusterIndex, const float tanLambda,
    const float directionZIntersection, const std::array<int, 4>& selectedBinsRect,
    const std::vector<std::pair<int, int>> &selectedClusters)
{

  const int rowsNum { static_cast<int>(selectedClusters.size()) };
  int maxClustersPerRow = 0;

  for (int iRow { 0 }; iRow < rowsNum; ++iRow) {

    if (selectedClusters[iRow].second > maxClustersPerRow) {

      maxClustersPerRow = selectedClusters[iRow].second;
    }
  }

  int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };
  if (phiBinsNum < 0) {

    phiBinsNum += CAConstants::IndexTable::PhiBins;
  }

  dim3 threadsPerBlock { getBlockSize(maxClustersPerRow, rowsNum) };
  dim3 blocksGrid { 1 + maxClustersPerRow / threadsPerBlock.x, 1 + rowsNum / threadsPerBlock.y };

  trackletsKernel<<< blocksGrid, threadsPerBlock, 0, streams[currentLayerIndex] >>>(primaryVertexContext.gpuContextDevicePointer,
      currentLayerIndex, currentClusterIndex, tanLambda, directionZIntersection,
      selectedBinsRect[0], selectedBinsRect[1], selectedBinsRect[2], phiBinsNum);

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {

    std::ostringstream errorString { };
    errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;

    throw std::runtime_error { errorString.str() };
  }
}
