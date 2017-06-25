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

#include <array>
#include <vector>

#include "CACell.h"
#include "CAConstants.h"
#include "CADefinitions.h"
#include "CAEvent.h"
#include "CAPrimaryVertexContextInitializer.h"
#include "CARoad.h"
#include "CATracklet.h"

using namespace TRACKINGITSU_TARGET_NAMESPACE;

  template<bool IsGPU>
  class CAPrimaryVertexContext
    final
    {
      public:
        explicit CAPrimaryVertexContext(const CAEvent&, const int);

        CAPrimaryVertexContext(const CAPrimaryVertexContext&) = delete;
        CAPrimaryVertexContext &operator=(const CAPrimaryVertexContext&) = delete;

        const float3& getPrimaryVertex();
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
        const float3 mPrimaryVertex;
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
    CAPrimaryVertexContext<IsGPU>::CAPrimaryVertexContext(const CAEvent& event, const int primaryVertexIndex)
        : mPrimaryVertex { event.getPrimaryVertex(primaryVertexIndex) }, mClusters {
            CAPrimaryVertexContextInitializer::initClusters(event, primaryVertexIndex) }, mIndexTables {
            CAPrimaryVertexContextInitializer::initIndexTables(mClusters) }, mTracklets {
            CAPrimaryVertexContextInitializer::initTracklets(event) }, mTrackletsLookupTable {
            CAPrimaryVertexContextInitializer::initTrackletsLookupTable(event) }, mCells {
            CAPrimaryVertexContextInitializer::initCells(event) }, mCellsLookupTable {
            CAPrimaryVertexContextInitializer::initCellsLookupTable(event) }
    {
      // Nothing to do
    }

    template<bool IsGPU>
    const float3& CAPrimaryVertexContext<IsGPU>::getPrimaryVertex()
    {
      return mPrimaryVertex;
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
