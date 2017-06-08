/// \file CAPrimaryVertexContext.h
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
#ifndef TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXT_H_
#define TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXT_H_

#include <algorithm>
#include <array>
#include <vector>

#include "CACell.h"
#include "CAConstants.h"
#include "CADefinitions.h"
#include "CAEvent.h"
#include "CARoad.h"
#include "CATracklet.h"

using namespace TRACKINGITSU_TARGET_NAMESPACE;

template<bool IsGPU>
struct CAPrimaryVertexContextInitializer
    final
    {
      static std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> initClusters(const CAEvent&, const int);
      static std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
          CAConstants::ITS::TrackletsPerRoad> initIndexTables(
          std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>&);
      static std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> initTracklets(const CAEvent&);
      static std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> initTrackletsLookupTable(const CAEvent&);
      static std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> initCells(const CAEvent&);
      static std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> initCellsLookupTable(const CAEvent&);
  };

  template<bool IsGPU>
  class CAPrimaryVertexContext
    final
    {
      public:
        explicit CAPrimaryVertexContext(const CAEvent&, const int);

        CAPrimaryVertexContext(const CAPrimaryVertexContext&) = delete;
        CAPrimaryVertexContext &operator=(const CAPrimaryVertexContext&) = delete;

        int getPrimaryVertex();
        std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>& getClusters();
        std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
            CAConstants::ITS::TrackletsPerRoad>& getIndexTables();
        std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& getTracklets();
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>& getTrackletsLookupTable();
        std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>& getCells();
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
        std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1>& getCellsNeighbours();
        std::vector<CARoad>& getRoads();

      private:
        const int mPrimaryVertexIndex;
        std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> mClusters;
        std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
            CAConstants::ITS::TrackletsPerRoad> mIndexTables;
        std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> mTracklets;
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> mTrackletsLookupTable;
        std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> mCells;
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> mCellsLookupTable;
        std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1> mCellsNeighbours;
        std::vector<CARoad> mRoads;
    };

    template<bool IsGPU>
    std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> CAPrimaryVertexContextInitializer<IsGPU>::initClusters(
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

    template<bool IsGPU>
    std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
        CAConstants::ITS::TrackletsPerRoad> CAPrimaryVertexContextInitializer<IsGPU>::initIndexTables(
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

        for (int iBin { previousBinIndex + 1 };
            iBin <= CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins; iBin++) {

          indexTables[iLayer - 1][iBin] = clustersNum;
        }
      }

      return indexTables;
    }

    template<bool IsGPU>
    std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> CAPrimaryVertexContextInitializer<IsGPU>::initTracklets(
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

    template<bool IsGPU>
    std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> CAPrimaryVertexContextInitializer<IsGPU>::initTrackletsLookupTable(
        const CAEvent &event)
    {
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> trackletsLookupTable;

      for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

        trackletsLookupTable[iLayer].resize(event.getLayer(iLayer + 1).getClustersSize(),
            CAConstants::ITS::UnusedIndex);
      }

      return trackletsLookupTable;
    }

    template<bool IsGPU>
    std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> CAPrimaryVertexContextInitializer<IsGPU>::initCells(
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

    template<bool IsGPU>
    std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> CAPrimaryVertexContextInitializer<IsGPU>::initCellsLookupTable(
        const CAEvent &event)
    {
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> cellsLookupTable;

      for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad - 1; ++iLayer) {

        cellsLookupTable[iLayer].resize(
            std::ceil(
                (CAConstants::Memory::TrackletsMemoryCoefficients[iLayer + 1]
                    * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize()),
            CAConstants::ITS::UnusedIndex);
      }

      return cellsLookupTable;
    }

    template<bool IsGPU>
    CAPrimaryVertexContext<IsGPU>::CAPrimaryVertexContext(const CAEvent& event, const int primaryVertexIndex)
        : mPrimaryVertexIndex { primaryVertexIndex }, mClusters {
            CAPrimaryVertexContextInitializer<IsGPU>::initClusters(event, primaryVertexIndex) }, mIndexTables {
            CAPrimaryVertexContextInitializer<IsGPU>::initIndexTables(mClusters) }, mTracklets {
            CAPrimaryVertexContextInitializer<IsGPU>::initTracklets(event) }, mTrackletsLookupTable {
            CAPrimaryVertexContextInitializer<IsGPU>::initTrackletsLookupTable(event) }, mCells {
            CAPrimaryVertexContextInitializer<IsGPU>::initCells(event) }, mCellsLookupTable {
            CAPrimaryVertexContextInitializer<IsGPU>::initCellsLookupTable(event) }
    {
      // Nothing to do
    }

    template<bool IsGPU>
    int CAPrimaryVertexContext<IsGPU>::getPrimaryVertex()
    {
      return mPrimaryVertexIndex;
    }

    template<bool IsGPU>
    std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>& CAPrimaryVertexContext<IsGPU>::getClusters()
    {
      return mClusters;
    }

    template<bool IsGPU>
    std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
        CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext<IsGPU>::getIndexTables()
    {
      return mIndexTables;
    }

    template<bool IsGPU>
    std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext<IsGPU>::getTracklets()
    {
      return mTracklets;
    }

    template<bool IsGPU>
    std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<IsGPU>::getTrackletsLookupTable()
    {
      return mTrackletsLookupTable;
    }

    template<bool IsGPU>
    std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext<IsGPU>::getCells()
    {
      return mCells;
    }

    template<bool IsGPU>
    std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext<IsGPU>::getCellsLookupTable()
    {
      return mCellsLookupTable;
    }

    template<bool IsGPU>
    std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext<IsGPU>::getCellsNeighbours()
    {
      return mCellsNeighbours;
    }

    template<bool IsGPU>
    std::vector<CARoad>& CAPrimaryVertexContext<IsGPU>::getRoads()
    {
      return mRoads;
    }

#endif /* TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXT_H_ */
