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
#include <cuda_runtime.h>

#include "CAConstants.h"
#include "CAGPUVector.h"
#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"
#include "CATrackingUtils.h"

namespace {
//TODO: this must be refined with runtime device queries or with careful planning
constexpr int WarpSize { 32 };
constexpr int MaxXThreads { 1024 };
constexpr int MaxThreadsPerBlock { 1024 };

dim3 getBlockSize(const int colsNum, const int rowsNum)
{
  const int xThreads { min(CAMathUtils::roundUp(rowsNum, WarpSize), MaxXThreads) };
  const int yThreads { colsNum * xThreads <= MaxThreadsPerBlock ? colsNum : MaxThreadsPerBlock / xThreads };

  return dim3 { static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads) };
}

__device__ int getWarpIndex()
{
  return (threadIdx.x + threadIdx.y * blockDim.x) / WarpSize;
}

__device__ int shareToWarp(int value, int leaderIndex)
{
  return __shfl(value, leaderIndex);
}

__global__ void trackletsKernel(CAGPUVector<CACluster> currentLayerClusters, CAGPUVector<CACluster> nextLayerClusters,
    CAGPUVector<CATracklet> tracklets, CAGPUVector<CAIndexTable> indexTables, CAGPUVector<int> trackletsLookupTable,
    const int firstClusterIndex, const float tanLambda, const float directionZIntersection, const int minZBinIndex,
    const int minPhiBinIndex, const int maxZBinIndex, const int maxPhiBinIndex)
{
  const CACluster& currentCluster { currentLayerClusters[firstClusterIndex] };
  const CAIndexTable& indexTable { indexTables[currentCluster.layerIndex] };
  const int phiBinIndex { (minPhiBinIndex + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y))
      % CAConstants::IndexTable::PhiBins };

  if (phiBinIndex >= minPhiBinIndex && phiBinIndex <= maxPhiBinIndex) {

    const int nextClusterIndex { indexTable.getBinFirstClusterIndex(
        CAIndexTableUtils::getBinIndex(minZBinIndex, phiBinIndex))
        + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
    const int maxClusterIndex { indexTable.getBinFirstClusterIndex(
        CAIndexTableUtils::getBinIndex(maxZBinIndex + 1, phiBinIndex)) };

    if (nextClusterIndex <= maxClusterIndex) {

      const CACluster& nextCluster { nextLayerClusters[nextClusterIndex] };

      CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection);

      const int warpIndex { getWarpIndex() };
      const unsigned int mask { __ballot(1) };
      const int leaderIndex { __ffs(mask) - 1 };
      int startIndex { };

      if (warpIndex == leaderIndex) {

        startIndex = tracklets.extend(__popc(mask));

        if (currentCluster.layerIndex > 0) {

          atomicCAS(trackletsLookupTable.get() + firstClusterIndex, CAConstants::ITS::UnusedIndex, startIndex);
        }
      }

      startIndex = shareToWarp(startIndex, leaderIndex);

      tracklets.insert(startIndex + __popc(mask & ((1 << warpIndex) - 1)), firstClusterIndex, nextClusterIndex,
          currentCluster, nextCluster);
    }
  }
}
}

void CAGPUTrackingAPI::getTrackletsFromCluster(CAPrimaryVertexContext& primaryVertexContext,
    const int currentLayerIndex, const int firstClusterIndex, const float tanLambda, const float directionZIntersection,
    const std::array<int, 4>& selectedBinsRect, const std::vector<std::pair<int, int>> &selectedClusters)
{
  const int rowsNum { static_cast<int>(selectedClusters.size()) };
  int maxClustersPerRow = 0;

  for (int iRow { 0 }; iRow < rowsNum; ++iRow) {

    if (selectedClusters[iRow].second > maxClustersPerRow) {

      maxClustersPerRow = selectedClusters[iRow].second;
    }
  }

  dim3 threadsPerBlock { getBlockSize(maxClustersPerRow, rowsNum) };
  dim3 blocksGrid { maxClustersPerRow / threadsPerBlock.x, rowsNum / threadsPerBlock.y };

  trackletsKernel<<< blocksGrid, threadsPerBlock >>>(primaryVertexContext.dClusters[currentLayerIndex],
      primaryVertexContext.dClusters[currentLayerIndex + 1], primaryVertexContext.dTracklets[currentLayerIndex], primaryVertexContext.dIndexTables,
      primaryVertexContext.dTrackletsLookupTable[currentLayerIndex], firstClusterIndex, tanLambda, directionZIntersection, selectedBinsRect[0], selectedBinsRect[1], selectedBinsRect[2], selectedBinsRect[3]);
}
