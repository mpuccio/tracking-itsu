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

__host__        __device__ dim3 getBlockSize(const int colsNum, const int rowsNum)
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

__host__        __device__ dim3 getBlockSize(const int colsNum)
{
  return getBlockSize(colsNum, 1);
}

__device__ int getLaneIndex()
{
  return (threadIdx.x + threadIdx.y * blockDim.x) % WarpSize;
}

__device__ int shareToWarp(int value, int laneIndex)
{
  return __shfl(value, laneIndex);
}

__device__ void computeLayerTracklets(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    float3 primaryVertex, int &clusterTracklets, bool dryRun)
{
  const int currentClusterIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int startIndex;
  int currentIndex = 0;

  if (!dryRun) {

    int laneIndex = getLaneIndex();

    if (laneIndex == WarpSize - 1) {

      startIndex = primaryVertexContext.getTracklets()[layerIndex].extend(clusterTracklets);
    }

    startIndex = shareToWarp(startIndex, WarpSize - 1);
    const int currentClusterOffset = __shfl_up(clusterTracklets, 1);

    if (laneIndex != 0) {

      startIndex += currentClusterOffset;
    }
  }

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

      const int nextLayerClustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };
      int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

      if (phiBinsNum < 0) {

        phiBinsNum += CAConstants::IndexTable::PhiBins;
      }

      if (!dryRun && layerIndex > 0) {

        primaryVertexContext.getTrackletsLookupTable()[layerIndex - 1][currentClusterIndex] = startIndex;
      }

      for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
          iPhiBin = ++iPhiBin == CAConstants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

        const int firstBinIndex { CAIndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };
        const int maxBinIndex { firstBinIndex + selectedBinsRect[2] - selectedBinsRect[0] + 1 };
        const int firstRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][firstBinIndex];
        const int maxRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][maxBinIndex];

        for (int iNextLayerCluster { firstRowClusterIndex };
            iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

          const CACluster& nextCluster { primaryVertexContext.getClusters()[layerIndex + 1][iNextLayerCluster] };

          if (CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

            if (dryRun) {

              ++clusterTracklets;

            } else {

              primaryVertexContext.getTracklets()[layerIndex].emplace(startIndex + currentIndex, currentClusterIndex,
                  iNextLayerCluster, currentCluster, nextCluster);
              ++currentIndex;
            }
          }
        }
      }
    }
  }
}

__device__ void computeLayerCells(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    float3 primaryVertex, int &trackletCells, bool dryRun)
{
  const int currentTrackletIndex = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int startIndex;
  int currentIndex = 0;

  if (!dryRun) {

    int laneIndex = getLaneIndex();

    if (laneIndex == WarpSize - 1) {

      startIndex = primaryVertexContext.getCells()[layerIndex].extend(trackletCells);
    }

    startIndex = shareToWarp(startIndex, WarpSize - 1);
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

    /*
     if (nextLayerFirstTrackletIndex == CAConstants::ITS::UnusedIndex) {

     continue;
     }
     */

    const CACluster& firstCellCluster {
        primaryVertexContext.getClusters()[layerIndex][currentTracklet.firstClusterIndex] };
    const CACluster& secondCellCluster {
        primaryVertexContext.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex] };
    const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
    const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
    const GPU_ARRAY<float, 3> firstDeltaVector { { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
        secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
            - firstCellClusterQuadraticRCoordinate } };
    const int nextLayerTrackletsNum { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].size()) };

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

          const float thirdCellClusterQuadraticRCoordinate { thirdCellCluster.rCoordinate * thirdCellCluster.rCoordinate };

          const GPU_ARRAY<float, 3> secondDeltaVector { { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
              thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate
                  - firstCellClusterQuadraticRCoordinate } };

          GPU_ARRAY<float, 3> cellPlaneNormalVector { CAMathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

          const float vectorNorm { std::sqrt(
              cellPlaneNormalVector[0] * cellPlaneNormalVector[0] + cellPlaneNormalVector[1] * cellPlaneNormalVector[1]
                  + cellPlaneNormalVector[2] * cellPlaneNormalVector[2]) };

          if (!(vectorNorm < CAConstants::Math::FloatMinThreshold
              || MATH_ABS(cellPlaneNormalVector[2]) < CAConstants::Math::FloatMinThreshold)) {

            const float inverseVectorNorm { 1.0f / vectorNorm };
            const GPU_ARRAY<float, 3> normalizedPlaneVector { { cellPlaneNormalVector[0] * inverseVectorNorm,
                cellPlaneNormalVector[1] * inverseVectorNorm, cellPlaneNormalVector[2] * inverseVectorNorm } };
            const float planeDistance { -normalizedPlaneVector[0] * (secondCellCluster.xCoordinate - primaryVertex.x)
                - (normalizedPlaneVector[1] * secondCellCluster.yCoordinate - primaryVertex.y)
                - normalizedPlaneVector[2] * secondCellClusterQuadraticRCoordinate };
            const float normalizedPlaneVectorQuadraticZCoordinate { normalizedPlaneVector[2] * normalizedPlaneVector[2] };
            const float cellTrajectoryRadius { MATH_SQRT(
                (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector[2])
                    / (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };
            const GPU_ARRAY<float, 2> circleCenter { { -0.5f * normalizedPlaneVector[0] / normalizedPlaneVector[2], -0.5f
                * normalizedPlaneVector[1] / normalizedPlaneVector[2] } };
            const float distanceOfClosestApproach { MATH_ABS(
                cellTrajectoryRadius - MATH_SQRT(circleCenter[0] * circleCenter[0] + circleCenter[1] * circleCenter[1])) };

            if (distanceOfClosestApproach
                <= CAConstants::Thresholds::CellMaxDistanceOfClosestApproachThreshold()[layerIndex]) {

              if (dryRun) {

                ++trackletCells;

              } else {

                const float cellTrajectoryCurvature { 1.0f / cellTrajectoryRadius };

                primaryVertexContext.getCells()[layerIndex].emplace(startIndex + currentIndex,
                    currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex,
                    currentTrackletIndex, iNextLayerTracklet, normalizedPlaneVector, cellTrajectoryCurvature);
                ++currentIndex;
              }
            }
          }
        }
      }
    }
  }
}

__global__ void layerTrackletsKernel(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    float3 primaryVertex)
{
  int clusterTracklets = 0;
  const int laneIndex = getLaneIndex();

  computeLayerTracklets(primaryVertexContext, layerIndex, primaryVertex, clusterTracklets, true);

  for (int iOffset = WarpSize / 2; iOffset > 0; iOffset /= 2) {

    int clustersToSum = __shfl_up(clusterTracklets, iOffset);

    if (laneIndex >= iOffset) {

      clusterTracklets += clustersToSum;
    }
  }

  computeLayerTracklets(primaryVertexContext, layerIndex, primaryVertex, clusterTracklets, false);
}

__global__ void layerCellsKernel(CAGPUPrimaryVertexContext& primaryVertexContext, const int layerIndex,
    float3 primaryVertex)
{
  int trackletCells = 0;
  const int laneIndex = getLaneIndex();

  computeLayerCells(primaryVertexContext, layerIndex, primaryVertex, trackletCells, true);

  for (int iOffset = WarpSize / 2; iOffset > 0; iOffset /= 2) {

    int trackletsToSum = __shfl_up(trackletCells, iOffset);

    if (laneIndex >= iOffset) {

      trackletCells += trackletsToSum;
    }
  }

  computeLayerCells(primaryVertexContext, layerIndex, primaryVertex, trackletCells, false);
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
    void computeLayerCells(Context&, const int, const std::array<float, 3>&);
    void postProcessCells(Context&);

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
      layerIndex, {primaryVertex[0], primaryVertex[1], primaryVertex[2]});

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
}

void CATrackerTraits<true>::computeLayerCells(Context& primaryVertexContext, const int layerIndex,
    const std::array<float, 3> &primaryVertex)
{
  const std::unique_ptr<int, void (*)(void*)> trackletsSizeUniquePointer =
      primaryVertexContext.getDeviceTracklets()[layerIndex].getSizeFromDevice();

  dim3 threadsPerBlock { getBlockSize(*trackletsSizeUniquePointer) };
  dim3 blocksGrid { 1 + *trackletsSizeUniquePointer / threadsPerBlock.x };

  cudaStream_t currentStream;
  cudaStreamCreate(&currentStream);

  layerCellsKernel<<< blocksGrid, threadsPerBlock, 0, currentStream >>>(primaryVertexContext.getDeviceContext(),
      layerIndex, {primaryVertex[0], primaryVertex[1], primaryVertex[2]});

  cudaError_t error = cudaGetLastError();

  cudaStreamDestroy(currentStream);

  if (error != cudaSuccess) {

    std::ostringstream errorString { };
    errorString << "CUDA API returned error [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;

    throw std::runtime_error { errorString.str() };
  }
}

void CATrackerTraits<true>::postProcessCells(Context& primaryVertexContext)
{
  cudaDeviceSynchronize();

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    const std::unique_ptr<int, void (*)(void*)> cellsSizeUniquePointer =
        primaryVertexContext.getDeviceCells()[iLayer].getSizeFromDevice();
    primaryVertexContext.getDeviceCells()[iLayer].copyIntoVector(primaryVertexContext.getCells()[iLayer],
        *cellsSizeUniquePointer);

    if (iLayer < CAConstants::ITS::CellsPerRoad - 1) {

      const std::unique_ptr<int, void (*)(void*)> trackletsSizeUniquePointer =
          primaryVertexContext.getDeviceTracklets()[iLayer + 1].getSizeFromDevice();
      primaryVertexContext.getDeviceCellsLookupTable()[iLayer].copyIntoVector(
          primaryVertexContext.getCellsLookupTable()[iLayer], *trackletsSizeUniquePointer);
    }
  }
}
