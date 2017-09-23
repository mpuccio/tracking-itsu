/// \file PrimaryVertexContext.cxx
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

#include "ITSReconstruction/CA/PrimaryVertexContext.h"

#include "ITSReconstruction/CA/Event.h"

namespace o2
{
namespace ITS
{
namespace CA
{

PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

void PrimaryVertexContext::initialize(const Event& event, const int primaryVertexIndex) {
  mPrimaryVertex = event.getPrimaryVertex(primaryVertexIndex);

  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const Layer& currentLayer { event.getLayer(iLayer) };
    const int clustersNum { currentLayer.getClustersSize() };

    mClusters[iLayer].clear();

    if(clustersNum > mClusters[iLayer].capacity()) {

      mClusters[iLayer].reserve(clustersNum);
    }

    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

      const Cluster& currentCluster { currentLayer.getCluster(iCluster) };
      mClusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
    }

    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });

    if(iLayer < Constants::ITS::CellsPerRoad) {

      mCells[iLayer].clear();
      float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
         * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());

      if(cellsMemorySize > mCells[iLayer].capacity()) {

        mCells[iLayer].reserve(cellsMemorySize);
      }
    }

    if(iLayer < Constants::ITS::CellsPerRoad - 1) {

      mCellsLookupTable[iLayer].clear();
      mCellsLookupTable[iLayer].resize(std::ceil(
        (Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
          * event.getLayer(iLayer + 2).getClustersSize()), Constants::ITS::UnusedIndex);


      mCellsNeighbours[iLayer].clear();
    }
  }

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {

    mCellsNeighbours[iLayer].clear();
  }

  mRoads.clear();

#if TRACKINGITSU_GPU_MODE
  mGPUContextDevicePointer = mGPUContext.initialize(mPrimaryVertex, mClusters, mCells, mCellsLookupTable);
#else
  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const int clustersNum = static_cast<int>(mClusters[iLayer].size());

    if(iLayer > 0) {

      int previousBinIndex { 0 };
      mIndexTables[iLayer - 1][0] = 0;

      for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

        const int currentBinIndex { mClusters[iLayer][iCluster].indexTableBinIndex };

        if (currentBinIndex > previousBinIndex) {

          for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

            mIndexTables[iLayer - 1][iBin] = iCluster;
          }

          previousBinIndex = currentBinIndex;
        }
      }

      for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
          iBin++) {

        mIndexTables[iLayer - 1][iBin] = clustersNum;
      }
    }

    if(iLayer < Constants::ITS::TrackletsPerRoad) {

      mTracklets[iLayer].clear();

      float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
         * event.getLayer(iLayer + 1).getClustersSize());

      if(trackletsMemorySize > mTracklets[iLayer].capacity()) {

        mTracklets[iLayer].reserve(trackletsMemorySize);
      }
    }

    if(iLayer < Constants::ITS::CellsPerRoad) {

      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(
         event.getLayer(iLayer + 1).getClustersSize(), Constants::ITS::UnusedIndex);
    }
  }
#endif
}

}
}
}
