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

#include "cub.cuh"

#include "CAConstants.h"
#include "CAGPUContext.h"
#include "CAGPUPrimaryVertexContext.h"
#include "CAGPUStream.h"
#include "CAGPUVector.h"
#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"
#include "CAPrimaryVertexContext.h"
#include "CATrackingUtils.h"

__device__ void computeLayerTracklets(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex)
{
  extern __shared__ CACluster sharedClusters[];
  const int currentClusterIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int clusterTrackletsNum = 0;

  if (currentClusterIndex < primaryVertexContext.getClusters()[layerIndex].size()) {

    sharedClusters[threadIdx.x] = primaryVertexContext.getClusters()[layerIndex][currentClusterIndex];
    const CACluster& currentCluster { sharedClusters[threadIdx.x] };

    /*if (mUsedClustersTable[currentCluster.clusterId] != CAConstants::ITS::UnusedIndex) {

     continue;
     }*/

    const float tanLambda { (currentCluster.zCoordinate - primaryVertexContext.getPrimaryVertex().z)
        / currentCluster.rCoordinate };
    const float directionZIntersection { tanLambda
        * ((CAConstants::ITS::LayersRCoordinate())[layerIndex + 1] - currentCluster.rCoordinate)
        + currentCluster.zCoordinate };

    const GPUArray<int, 4> selectedBinsRect { CATrackingUtils::getBinsRect(currentCluster, layerIndex,
        directionZIntersection) };

    if (selectedBinsRect[0] != 0 || selectedBinsRect[1] != 0 || selectedBinsRect[2] != 0 || selectedBinsRect[3] != 0) {

      const int nextLayerClustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };
      int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

      if (phiBinsNum < 0) {

        phiBinsNum += CAConstants::IndexTable::PhiBins;
      }

      for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
          iPhiBin = ++iPhiBin == CAConstants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex { CAIndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };
        const int maxBinIndex { firstBinIndex + selectedBinsRect[2] - selectedBinsRect[0] + 1 };
        const int firstRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][firstBinIndex];
        const int maxRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][ { firstBinIndex
            + selectedBinsRect[2] - selectedBinsRect[0] + 1 }];

        for (int iNextLayerCluster { firstRowClusterIndex };
            iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

          const CACluster& nextCluster { primaryVertexContext.getClusters()[layerIndex + 1][iNextLayerCluster] };

          if (CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

            int mask { static_cast<int>(__ballot(1)) };
            int leader { __ffs(mask) - 1 };
            int laneIndex { CAGPUUtils::Device::getLaneIndex() };
            int startIndex { };

            if (laneIndex == leader) {

              startIndex = primaryVertexContext.getTracklets()[layerIndex].extend(__popc(mask));
            }

            startIndex = CAGPUUtils::Device::shareToWarp(startIndex, leader) + __popc(mask & ((1 << laneIndex) - 1));

            primaryVertexContext.getTracklets()[layerIndex].emplace(startIndex, currentClusterIndex, iNextLayerCluster,
                currentCluster, nextCluster);
            ++clusterTrackletsNum;
          }
        }
      }

      if (layerIndex > 0) {

        primaryVertexContext.getTrackletsPerClusterTable()[layerIndex - 1][currentClusterIndex] = clusterTrackletsNum;
      }
    }
  }
}

__device__ void computeLayerCells(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    int &trackletCells, const int warpSize, const bool dryRun)
{
  const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const float3 &primaryVertex = primaryVertexContext.getPrimaryVertex();
  int startIndex;
  int currentIndex = 0;

  if (!dryRun) {

    int laneIndex = CAGPUUtils::Device::getLaneIndex();

    if (laneIndex == warpSize - 1) {

      startIndex = primaryVertexContext.getCells()[layerIndex].extend(trackletCells);
    }

    startIndex = CAGPUUtils::Device::shareToWarp(startIndex, warpSize - 1);
    const int currentTrackletOffset = __shfl_up(trackletCells, 1);

    if (laneIndex != 0) {

      startIndex += currentTrackletOffset;
    }
  }

  if (currentTrackletIndex < primaryVertexContext.getTracklets()[layerIndex].size()) {

    const CATracklet& currentTracklet { primaryVertexContext.getTracklets()[layerIndex][currentTrackletIndex] };
    const int nextLayerClusterIndex { currentTracklet.secondClusterIndex };
    const int nextLayerFirstTrackletIndex {
        primaryVertexContext.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex] };
    const int nextLayerTrackletsNum { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].size()) };

    if (nextLayerClusterIndex + 1 == nextLayerTrackletsNum
        || nextLayerFirstTrackletIndex
            != primaryVertexContext.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex + 1]) {

      const CACluster& firstCellCluster {
          primaryVertexContext.getClusters()[layerIndex][currentTracklet.firstClusterIndex] };
      const CACluster& secondCellCluster {
          primaryVertexContext.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex] };
      const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
      const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
      const float3 firstDeltaVector { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
          secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
              - firstCellClusterQuadraticRCoordinate };

      if (!dryRun && layerIndex > 0) {

        primaryVertexContext.getCellsLookupTable()[layerIndex - 1][currentTrackletIndex] = startIndex;
      }

      for (int iNextLayerTracklet { nextLayerFirstTrackletIndex };
          iNextLayerTracklet < nextLayerTrackletsNum
              && primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex
                  == nextLayerClusterIndex; ++iNextLayerTracklet) {

        const CATracklet& nextTracklet { primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet] };
        const float deltaTanLambda { MATH_ABS(currentTracklet.tanLambda - nextTracklet.tanLambda) };
        const float deltaPhi { MATH_ABS(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate) };

        if (deltaTanLambda < CAConstants::Thresholds::CellMaxDeltaTanLambdaThreshold
            && (deltaPhi < CAConstants::Thresholds::CellMaxDeltaPhiThreshold
                || MATH_ABS(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::CellMaxDeltaPhiThreshold)) {

          const float averageTanLambda { 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
          const float directionZIntersection { -averageTanLambda * firstCellCluster.rCoordinate
              + firstCellCluster.zCoordinate };
          const float deltaZ { MATH_ABS(directionZIntersection - primaryVertex.z) };

          if (deltaZ < CAConstants::Thresholds::CellMaxDeltaZThreshold()[layerIndex]) {

            const CACluster& thirdCellCluster {
                primaryVertexContext.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex] };

            const float thirdCellClusterQuadraticRCoordinate { thirdCellCluster.rCoordinate
                * thirdCellCluster.rCoordinate };

            const float3 secondDeltaVector { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate
                    - firstCellClusterQuadraticRCoordinate };

            float3 cellPlaneNormalVector { CAMathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

            const float vectorNorm { std::sqrt(
                cellPlaneNormalVector.x * cellPlaneNormalVector.x + cellPlaneNormalVector.y * cellPlaneNormalVector.y
                    + cellPlaneNormalVector.z * cellPlaneNormalVector.z) };

            if (!(vectorNorm < CAConstants::Math::FloatMinThreshold
                || MATH_ABS(cellPlaneNormalVector.z) < CAConstants::Math::FloatMinThreshold)) {

              const float inverseVectorNorm { 1.0f / vectorNorm };
              const float3 normalizedPlaneVector { cellPlaneNormalVector.x * inverseVectorNorm, cellPlaneNormalVector.y
                  * inverseVectorNorm, cellPlaneNormalVector.z * inverseVectorNorm };
              const float planeDistance { -normalizedPlaneVector.x * (secondCellCluster.xCoordinate - primaryVertex.x)
                  - (normalizedPlaneVector.y * secondCellCluster.yCoordinate - primaryVertex.y)
                  - normalizedPlaneVector.z * secondCellClusterQuadraticRCoordinate };
              const float normalizedPlaneVectorQuadraticZCoordinate { normalizedPlaneVector.z * normalizedPlaneVector.z };
              const float cellTrajectoryRadius { MATH_SQRT(
                  (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector.z)
                      / (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };
              const float2 circleCenter { -0.5f * normalizedPlaneVector.x / normalizedPlaneVector.z, -0.5f
                  * normalizedPlaneVector.y / normalizedPlaneVector.z };
              const float distanceOfClosestApproach { MATH_ABS(
                  cellTrajectoryRadius - MATH_SQRT(circleCenter.x * circleCenter.x + circleCenter.y * circleCenter.y)) };

              if (distanceOfClosestApproach
                  <= CAConstants::Thresholds::CellMaxDistanceOfClosestApproachThreshold()[layerIndex]) {

                if (dryRun) {

                  ++trackletCells;

                } else {

                  primaryVertexContext.getCells()[layerIndex].emplace(startIndex + currentIndex,
                      currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex,
                      nextTracklet.secondClusterIndex, currentTrackletIndex, iNextLayerTracklet, normalizedPlaneVector,
                      1.0f / cellTrajectoryRadius);
                  ++currentIndex;
                }
              }
            }
          }
        }
      }
    }
  }
}

__global__ void layerTrackletsKernel(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex)
{
  computeLayerTracklets(primaryVertexContext, layerIndex);
}

__global__ void sortTrackletsKernel(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    CATracklet *tempTrackletArray)
{
  const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);

  if (currentTrackletIndex < primaryVertexContext.getTracklets()[layerIndex].size()) {

    const int firstClusterIndex =
        primaryVertexContext.getTracklets()[layerIndex][currentTrackletIndex].firstClusterIndex;
    const int offset = atomicAdd(&primaryVertexContext.getTrackletsPerClusterTable()[layerIndex - 1][firstClusterIndex],
        -1) - 1;
    const int startIndex = primaryVertexContext.getTrackletsLookupTable()[layerIndex - 1][firstClusterIndex];

    memcpy(&tempTrackletArray[startIndex + offset],
        &primaryVertexContext.getTracklets()[layerIndex][currentTrackletIndex], sizeof(CATracklet));
  }
}

__global__ void layerCellsKernel(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    const int warpSize)
{
  int trackletCells = 0;
  const int laneIndex = CAGPUUtils::Device::getLaneIndex();

  computeLayerCells(primaryVertexContext, layerIndex, trackletCells, warpSize, true);

  for (int iOffset = warpSize / 2; iOffset > 0; iOffset /= 2) {

    int trackletsToSum = __shfl_up(trackletCells, iOffset);

    if (laneIndex >= iOffset) {

      trackletCells += trackletsToSum;
    }
  }

  computeLayerCells(primaryVertexContext, layerIndex, trackletCells, warpSize, false);
}

template<>
void CATrackerTraits<true>::computeLayerTracklets(Context& primaryVertexContext)
{
  std::array<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad> tempTableArray;
  std::array<size_t, CAConstants::ITS::CellsPerRoad> tempSize;
  std::array<CAGPUVector<CATracklet>, CAConstants::ITS::CellsPerRoad> tempTrackletArray;
  std::array<int, CAConstants::ITS::CellsPerRoad> trackletsNum;
  std::array<CAGPUStream, CAConstants::ITS::TrackletsPerRoad> streamArray;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    tempSize[iLayer] = 0;
    const int trackletsNum { static_cast<int>(primaryVertexContext.getDeviceTracklets()[iLayer + 1].capacity()) };
    tempTrackletArray[iLayer] = CAGPUVector<CATracklet> { trackletsNum };

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(NULL), tempSize[iLayer],
        primaryVertexContext.getDeviceTrackletsPerClustersTable()[iLayer].get(),
        primaryVertexContext.getDeviceTrackletsLookupTable()[iLayer].get(),
        primaryVertexContext.getClusters()[iLayer + 1].size());

    tempTableArray[iLayer] = CAGPUVector<int> { static_cast<int>(tempSize[iLayer]) };
  }

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const CAGPUDeviceProperties& deviceProperties = CAGPUContext::getInstance().getDeviceProperties();
    const int clustersNum { static_cast<int>(primaryVertexContext.getClusters()[iLayer].size()) };
    dim3 threadsPerBlock { CAGPUUtils::Host::getBlockSize(clustersNum) };
    dim3 blocksGrid { CAGPUUtils::Host::getBlocksGrid(threadsPerBlock, clustersNum) };

    layerTrackletsKernel<<< blocksGrid, threadsPerBlock, threadsPerBlock.x * sizeof(CACluster), streamArray[iLayer].get() >>>(primaryVertexContext.getDeviceContext(),
        iLayer);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    cudaStreamSynchronize(streamArray[iLayer + 1].get());
    trackletsNum[iLayer] = *primaryVertexContext.getDeviceTracklets()[iLayer + 1].getSizeFromDevice();

    cub::DeviceScan::ExclusiveSum(static_cast<void *>(tempTableArray[iLayer].get()), tempSize[iLayer],
        primaryVertexContext.getDeviceTrackletsPerClustersTable()[iLayer].get(),
        primaryVertexContext.getDeviceTrackletsLookupTable()[iLayer].get(),
        primaryVertexContext.getClusters()[iLayer + 1].size(), streamArray[iLayer + 1].get());

    dim3 threadsPerBlock { CAGPUUtils::Host::getBlockSize(trackletsNum[iLayer]) };
    dim3 blocksGrid { CAGPUUtils::Host::getBlocksGrid(threadsPerBlock, trackletsNum[iLayer]) };

    sortTrackletsKernel<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer + 1].get() >>>(primaryVertexContext.getDeviceContext(),
        iLayer + 1, tempTrackletArray[iLayer].get());

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")"
          << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    cudaStreamSynchronize(streamArray[iLayer + 1].get());

    tempTrackletArray[iLayer].resize(trackletsNum[iLayer]);
    primaryVertexContext.getDeviceTracklets()[iLayer + 1] = std::move(tempTrackletArray[iLayer]);
  }

  primaryVertexContext.updateDeviceContext();
}

template<>
void CATrackerTraits<true>::computeLayerCells(Context& primaryVertexContext, const int layerIndex)
{
  const CAGPUDeviceProperties& deviceProperties = CAGPUContext::getInstance().getDeviceProperties();
  const std::unique_ptr<int, void (*)(void *)> trackletsSize =
      primaryVertexContext.getDeviceTracklets()[layerIndex].getSizeFromDevice();

  dim3 threadsPerBlock { CAGPUUtils::Host::getBlockSize(*trackletsSize) };
  dim3 blocksGrid { CAGPUUtils::Host::getBlocksGrid(threadsPerBlock, *trackletsSize) };

  CAGPUStream stream { };

  layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, stream.get() >>>(primaryVertexContext.getDeviceContext(),
      layerIndex, deviceProperties.warpSize);

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {

    std::ostringstream errorString { };
    errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;

    throw std::runtime_error { errorString.str() };
  }
}

template<>
void CATrackerTraits<true>::postProcessCells(Context& primaryVertexContext)
{
  cudaDeviceSynchronize();

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    const std::unique_ptr<int, void (*)(void *)> cellsSize =
        primaryVertexContext.getDeviceCells()[iLayer].getSizeFromDevice();
    primaryVertexContext.getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext.getCells()[iLayer], *cellsSize);

    if (iLayer < CAConstants::ITS::CellsPerRoad - 1) {

      const std::unique_ptr<int, void (*)(void *)> cellsLookupTableSize =
          primaryVertexContext.getDeviceCellsLookupTable()[iLayer].getSizeFromDevice();
      primaryVertexContext.getDeviceCellsLookupTable()[iLayer].copyIntoVector(
          primaryVertexContext.getCellsLookupTable()[iLayer], *cellsLookupTableSize);
    }
  }
}
