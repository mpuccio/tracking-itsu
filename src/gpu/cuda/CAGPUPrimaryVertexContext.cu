/// \file CAGPUPrimaryVertexContext.cxx
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

#include "CAGPUPrimaryVertexContext.h"

#include <sstream>

#include "CAGPUStream.h"

namespace {
__device__ void fillIndexTables(CAGPUPrimaryVertexContext &primaryVertexContext, const int layerIndex)
{

  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int nextLayerClustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < nextLayerClustersNum) {

    const int currentBinIndex {
        primaryVertexContext.getClusters()[layerIndex + 1][currentClusterIndex].indexTableBinIndex };
    int previousBinIndex;

    if (currentClusterIndex == 0) {

      primaryVertexContext.getIndexTables()[layerIndex][0] = 0;
      previousBinIndex = 0;

    } else {

      previousBinIndex = primaryVertexContext.getClusters()[layerIndex + 1][currentClusterIndex - 1].indexTableBinIndex;
    }

    if (currentBinIndex > previousBinIndex) {

      for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = currentClusterIndex;
      }

      previousBinIndex = currentBinIndex;
    }

    if (currentClusterIndex == nextLayerClustersNum - 1) {

      for (int iBin { currentBinIndex + 1 }; iBin <= CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins;
          iBin++) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = nextLayerClustersNum;
      }
    }
  }
}

__device__ void fillTrackletsPerClusterTables(CAGPUPrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int clustersSize { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < clustersSize) {

    primaryVertexContext.getTrackletsPerClusterTable()[layerIndex][currentClusterIndex] = 0;
  }
}

__device__ void fillCellsPerClusterTables(CAGPUPrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  const int totalThreadNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };
  const int trackletsSize { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].capacity()) };
  const int trackletsPerThread { 1 + (trackletsSize - 1) / totalThreadNum };
  const int firstTrackletIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) * trackletsPerThread };

  if (firstTrackletIndex < trackletsSize) {

    const int trackletsToSet { min(trackletsSize, firstTrackletIndex + trackletsPerThread) - firstTrackletIndex };
    memset(&primaryVertexContext.getCellsPerTrackletTable()[layerIndex][firstTrackletIndex], 0,
        trackletsToSet * sizeof(int));
  }
}

__global__ void fillDeviceStructures(CAGPUPrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  fillIndexTables(primaryVertexContext, layerIndex);

  if (layerIndex < CAConstants::ITS::CellsPerRoad) {

    fillTrackletsPerClusterTables(primaryVertexContext, layerIndex);
  }

  if (layerIndex < CAConstants::ITS::CellsPerRoad - 1) {

    fillCellsPerClusterTables(primaryVertexContext, layerIndex);
  }
}
}

CAGPUPrimaryVertexContext::CAGPUPrimaryVertexContext(const float3 &primaryVertex,
    const std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> &clusters,
    const std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> &cells,
    const std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> &cellsLookupTable)
    : mPrimaryVertex { primaryVertex }
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    this->mClusters[iLayer] =
        CAGPUVector<CACluster> { &clusters[iLayer][0], static_cast<int>(clusters[iLayer].size()) };

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      this->mTracklets[iLayer] = CAGPUVector<CATracklet> { static_cast<int>(std::ceil(
          (CAConstants::Memory::TrackletsMemoryCoefficients[iLayer] * clusters[iLayer].size())
              * clusters[iLayer + 1].size())) };
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      this->mTrackletsLookupTable[iLayer] = CAGPUVector<int> { static_cast<int>(clusters[iLayer + 1].size()) };
      this->mTrackletsPerClusterTable[iLayer] = CAGPUVector<int> { static_cast<int>(clusters[iLayer + 1].size()) };
      this->mCells[iLayer] = CAGPUVector<CACell> { static_cast<int>(cells[iLayer].capacity()) };
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad - 1) {

      this->mCellsLookupTable[iLayer] = CAGPUVector<int> { static_cast<int>(cellsLookupTable[iLayer].size()) };
      this->mCellsPerTrackletTable[iLayer] = CAGPUVector<int> { static_cast<int>(cellsLookupTable[iLayer].size()) };
    }
  }
}

CAPrimaryVertexContext<true>::CAPrimaryVertexContext(const CAEvent& event, const int primaryVertexIndex)
    : mPrimaryVertex { event.getPrimaryVertex(primaryVertexIndex) }, mClusters {
        CAPrimaryVertexContextInitializer::initClusters(event, primaryVertexIndex) }, mCells {
        CAPrimaryVertexContextInitializer::initCells(event) }, mCellsLookupTable {
        CAPrimaryVertexContextInitializer::initCellsLookupTable(event) }, mGPUContext { mPrimaryVertex, mClusters,
        mCells, mCellsLookupTable }, mGPUContextDevicePointer { mGPUContext }
{
  std::array<CAGPUStream, CAConstants::ITS::LayersNumber> streamArray;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const int nextLayerClustersNum = static_cast<int>(mClusters[iLayer + 1].size());

    dim3 threadsPerBlock { CAGPUUtils::Host::getBlockSize(nextLayerClustersNum) };
    dim3 blocksGrid { CAGPUUtils::Host::getBlocksGrid(threadsPerBlock, nextLayerClustersNum) };

    fillDeviceStructures<<< blocksGrid, threadsPerBlock, 0, streamArray[iLayer].get() >>>(*mGPUContextDevicePointer, iLayer);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {

      std::ostringstream errorString { };
      errorString << __FILE__ << ":" << __LINE__ << " CUDA API returned error [" << cudaGetErrorString(error)
          << "] (code " << error << ")" << std::endl;

      throw std::runtime_error { errorString.str() };
    }
  }
}

const float3& CAPrimaryVertexContext<true>::getPrimaryVertex()
{
  return mPrimaryVertex;
}

std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>& CAPrimaryVertexContext<true>::getClusters()
{
  return mClusters;
}

std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext<true>::getTracklets()
{
  return mTracklets;
}

std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<true>::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}

std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<true>::getCells()
{
  return mCells;
}

std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext<true>::getCellsLookupTable()
{
  return mCellsLookupTable;
}

std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext<true>::getCellsNeighbours()
{
  return mCellsNeighbours;
}

std::vector<CARoad>& CAPrimaryVertexContext<true>::getRoads()
{
  return mRoads;
}

CAGPUPrimaryVertexContext& CAPrimaryVertexContext<true>::getDeviceContext()
{
  return *mGPUContextDevicePointer;
}

CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber>& CAPrimaryVertexContext<true>::getDeviceClusters()
{
  return mGPUContext.getClusters();
}

CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext<true>::getDeviceTracklets()
{
  return mGPUContext.getTracklets();
}

CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<true>::getDeviceTrackletsLookupTable()
{
  return mGPUContext.getTrackletsLookupTable();
}

CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<true>::getDeviceTrackletsPerClustersTable()
{
  return mGPUContext.getTrackletsPerClusterTable();
}

CAGPUArray<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<true>::getDeviceCells()
{
  return mGPUContext.getCells();
}

CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext<true>::getDeviceCellsLookupTable()
{
  return mGPUContext.getCellsLookupTable();
}

CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext<true>::getDeviceCellsPerTrackletTable()
{
  return mGPUContext.getCellsPerTrackletTable();
}

void CAPrimaryVertexContext<true>::updateDeviceContext()
{
  mGPUContextDevicePointer = CAGPUUniquePointer<CAGPUPrimaryVertexContext> { mGPUContext };
}
