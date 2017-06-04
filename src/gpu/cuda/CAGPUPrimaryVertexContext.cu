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

namespace {
__global__ void fillIndexTables(CAGPUPrimaryVertexContext *primaryVertexContext)
{
  const int iLayer = threadIdx.x;

  const int layerClustersNum { static_cast<int>(primaryVertexContext->getClusters()[iLayer + 1].size()) };
  int previousBinIndex { 0 };
  primaryVertexContext->getIndexTables()[iLayer] = CAGPUArray<int,
      CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1> { };
  primaryVertexContext->getIndexTables()[iLayer][0] = 0;

  for (int iCluster { 0 }; iCluster < layerClustersNum; ++iCluster) {

    const int currentBinIndex { primaryVertexContext->getClusters()[iLayer + 1][iCluster].indexTableBinIndex };

    if (currentBinIndex > previousBinIndex) {

      for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

        primaryVertexContext->getIndexTables()[iLayer][iBin] = iCluster;
      }

      previousBinIndex = currentBinIndex;
    }
  }

  for (int iBin { previousBinIndex + 1 }; iBin <= CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins;
      iBin++) {

    primaryVertexContext->getIndexTables()[iLayer][iBin] = layerClustersNum;
  }
}
}

CAGPUPrimaryVertexContext::CAGPUPrimaryVertexContext(
    const std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> &clusters,
    const std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> &tracklets,
    const std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> &trackletsLookupTable)
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    this->mClusters[iLayer] = CAGPUVector<CACluster> { &clusters[iLayer][0], static_cast<int>(clusters[iLayer].size()) };

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      this->mTracklets[iLayer] = CAGPUVector<CATracklet> { static_cast<int>(tracklets[iLayer].capacity()) };
      this->mTrackletsPerClusterTable[iLayer] = CAGPUVector<int> { static_cast<int>(clusters[iLayer].size()) };
      this->mTrackletsPerClusterTable[iLayer].fill(0);
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      this->mTrackletsLookupTable[iLayer] = CAGPUVector<int> { &trackletsLookupTable[iLayer][0],
          static_cast<int>(trackletsLookupTable[iLayer].size()) };
    }
  }
}

CAGPUPrimaryVertexContext::~CAGPUPrimaryVertexContext()
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    mClusters[iLayer].destroy();

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      mTracklets[iLayer].destroy();
      mTrackletsPerClusterTable[iLayer].destroy();
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      mTrackletsLookupTable[iLayer].destroy();
    }
  }
}

GPU_HOST_DEVICE CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber>& CAGPUPrimaryVertexContext::getClusters()
{
  return mClusters;
}

GPU_DEVICE CAGPUArray<CAGPUArray<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
  CAConstants::ITS::TrackletsPerRoad>& CAGPUPrimaryVertexContext::getIndexTables()
{
  return mIndexTables;
}

GPU_HOST_DEVICE CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAGPUPrimaryVertexContext::getTracklets()
{
  return mTracklets;
}

GPU_HOST_DEVICE CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAGPUPrimaryVertexContext::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}

GPU_DEVICE CAGPUArray<CAGPUVector<int>, CAConstants::ITS::TrackletsPerRoad>& CAGPUPrimaryVertexContext::getTrackletsPerClusterTable()
{
  return mTrackletsPerClusterTable;
}

CAPrimaryVertexContext<true>::CAPrimaryVertexContext(const CAEvent& event, const int primaryVertexIndex)
    : mPrimaryVertexIndex { primaryVertexIndex }, mClusters { CAPrimaryVertexContextInitializer<true>::initClusters(event,
        primaryVertexIndex) }, mTracklets { CAPrimaryVertexContextInitializer<true>::initTracklets(event) }, mTrackletsLookupTable {
        CAPrimaryVertexContextInitializer<true>::initTrackletsLookupTable(event) }, mCells {
        CAPrimaryVertexContextInitializer<true>::initCells(event) }, mGPUContext { mClusters, mTracklets, mTrackletsLookupTable }
{
  try {

    CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void**>(&mGPUContextDevicePointer), sizeof(CAGPUPrimaryVertexContext));
    CAGPUUtils::Host::gpuMemcpyHostToDevice(mGPUContextDevicePointer, &mGPUContext, sizeof(CAGPUPrimaryVertexContext));

    dim3 threadsPerBlock { CAConstants::ITS::TrackletsPerRoad };

    fillIndexTables<<< 1, threadsPerBlock >>>(mGPUContextDevicePointer);
    cudaDeviceSynchronize();

  } catch (...) {

    CAGPUUtils::Host::gpuFree(mGPUContextDevicePointer);
  }
}

CAPrimaryVertexContext<true>::~CAPrimaryVertexContext()
{
  CAGPUUtils::Host::gpuFree (mGPUContextDevicePointer);
}

int CAPrimaryVertexContext<true>::getPrimaryVertex()
{
  return mPrimaryVertexIndex;
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
