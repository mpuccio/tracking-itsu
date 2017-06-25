/// \file CAPrimaryVertexContextInitializer.cxx
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

#include "CAPrimaryVertexContextInitializer.h"

#include <algorithm>
#include <cmath>

std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> CAPrimaryVertexContextInitializer::initClusters(
    const CAEvent& event, const int primaryVertexIndex)
{
  std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> clusters;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    const CALayer& currentLayer { event.getLayer(iLayer) };
    const int clustersNum { currentLayer.getClustersSize() };

    clusters[iLayer].reserve(clustersNum);

    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

      const CACluster& currentCluster { currentLayer.getCluster(iCluster) };
      clusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
    }

    std::sort(clusters[iLayer].begin(), clusters[iLayer].end(), [](CACluster& cluster1, CACluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });
  }

  return clusters;
}

std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
    CAConstants::ITS::TrackletsPerRoad> CAPrimaryVertexContextInitializer::initIndexTables(
    std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> &clusters)
{
  std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
      CAConstants::ITS::TrackletsPerRoad> indexTables;

  for (int iLayer { 1 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    const int clustersNum = static_cast<int>(clusters[iLayer].size());
    int previousBinIndex { 0 };
    indexTables[iLayer - 1][0] = 0;

    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

      const int currentBinIndex { clusters[iLayer][iCluster].indexTableBinIndex };

      if (currentBinIndex > previousBinIndex) {

        for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

          indexTables[iLayer - 1][iBin] = iCluster;
        }

        previousBinIndex = currentBinIndex;
      }
    }

    for (int iBin { previousBinIndex + 1 }; iBin <= CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins;
        iBin++) {

      indexTables[iLayer - 1][iBin] = clustersNum;
    }
  }

  return indexTables;
}

std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> CAPrimaryVertexContextInitializer::initTracklets(
    const CAEvent &event)
{
  std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> tracklets;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    tracklets[iLayer].reserve(
        std::ceil(
            (CAConstants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
                * event.getLayer(iLayer + 1).getClustersSize()));
  }

  return tracklets;
}

std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> CAPrimaryVertexContextInitializer::initTrackletsLookupTable(
    const CAEvent &event)
{
  std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> trackletsLookupTable;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    trackletsLookupTable[iLayer].resize(event.getLayer(iLayer + 1).getClustersSize(), CAConstants::ITS::UnusedIndex);
  }

  return trackletsLookupTable;
}

std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> CAPrimaryVertexContextInitializer::initCells(
    const CAEvent &event)
{
  std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> cells;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    cells[iLayer].reserve(
        std::ceil(
            ((CAConstants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
                * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize()));
  }

  return cells;
}

std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> CAPrimaryVertexContextInitializer::initCellsLookupTable(
    const CAEvent &event)
{
  std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> cellsLookupTable;

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad - 1; ++iLayer) {

    cellsLookupTable[iLayer].resize(
        std::ceil(
            (CAConstants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
                * event.getLayer(iLayer + 2).getClustersSize()), CAConstants::ITS::UnusedIndex);
  }

  return cellsLookupTable;
}
