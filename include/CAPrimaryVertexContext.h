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
#include <iostream>
#include <vector>

#include "CACell.h"
#include "CAConstants.h"
#include "CADefinitions.h"
#include "CAEvent.h"
#include "CARoad.h"
#include "CATracklet.h"

#if TRACKINGITSU_GPU_MODE
# include "CAGPUPrimaryVertexContext.h"
# include "CAGPUUniquePointer.h"
#endif

  class CAPrimaryVertexContext
    final
    {
      public:
        CAPrimaryVertexContext();

        CAPrimaryVertexContext(const CAPrimaryVertexContext&) = delete;
        CAPrimaryVertexContext &operator=(const CAPrimaryVertexContext&) = delete;

        void initialize(const CAEvent&, const int);
        const float3& getPrimaryVertex() const;
        std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>& getClusters();
        std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>& getCells();
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
        std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1>& getCellsNeighbours();
        std::vector<CARoad>& getRoads();

#if TRACKINGITSU_GPU_MODE
        CAGPUPrimaryVertexContext& getDeviceContext();
        CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber>& getDeviceClusters();
        CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& getDeviceTracklets();
        CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& getDeviceTrackletsLookupTable();
        CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& getDeviceTrackletsPerClustersTable();
        CAGPUArray<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad>& getDeviceCells();
        CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& getDeviceCellsLookupTable();
        CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& getDeviceCellsPerTrackletTable();
        std::array<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& getTempTableArray();
        std::array<CAGPUVector<CATracklet>, CAConstants::ITS::CellsPerRoad>& getTempTrackletArray();
        std::array<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad - 1>& getTempCellArray();
        void updateDeviceContext();
#else
        std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
            CAConstants::ITS::TrackletsPerRoad>& getIndexTables();
        std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& getTracklets();
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>& getTrackletsLookupTable();
#endif

      private:
        float3 mPrimaryVertex;
        std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> mClusters;
        std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> mCells;
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> mCellsLookupTable;
        std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1> mCellsNeighbours;
        std::vector<CARoad> mRoads;

#if TRACKINGITSU_GPU_MODE
        CAGPUPrimaryVertexContext mGPUContext;
        CAGPUUniquePointer<CAGPUPrimaryVertexContext> mGPUContextDevicePointer;
        std::array<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad> mTempTableArray;
        std::array<CAGPUVector<CATracklet>, CAConstants::ITS::CellsPerRoad> mTempTrackletArray;
        std::array<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad - 1> mTempCellArray;
#else
        std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
            CAConstants::ITS::TrackletsPerRoad> mIndexTables;
        std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> mTracklets;
        std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> mTrackletsLookupTable;
#endif
    };

    inline const float3& CAPrimaryVertexContext::getPrimaryVertex() const
    {
      return mPrimaryVertex;
    }

    inline std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>& CAPrimaryVertexContext::getClusters()
    {
      return mClusters;
    }

    inline std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getCells()
    {
      return mCells;
    }

    inline std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext::getCellsLookupTable()
    {
      return mCellsLookupTable;
    }

    inline std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext::getCellsNeighbours()
    {
      return mCellsNeighbours;
    }

    inline std::vector<CARoad>& CAPrimaryVertexContext::getRoads()
    {
      return mRoads;
    }

#if TRACKINGITSU_GPU_MODE
    inline CAGPUPrimaryVertexContext& CAPrimaryVertexContext::getDeviceContext()
    {
      return *mGPUContextDevicePointer;
    }

    inline CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber>& CAPrimaryVertexContext::getDeviceClusters()
    {
      return mGPUContext.getClusters();
    }

    inline CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext::getDeviceTracklets()
    {
      return mGPUContext.getTracklets();
    }

    inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getDeviceTrackletsLookupTable()
    {
      return mGPUContext.getTrackletsLookupTable();
    }

    inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getDeviceTrackletsPerClustersTable()
    {
      return mGPUContext.getTrackletsPerClusterTable();
    }

    inline CAGPUArray<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getDeviceCells()
    {
      return mGPUContext.getCells();
    }

    inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext::getDeviceCellsLookupTable()
    {
      return mGPUContext.getCellsLookupTable();
    }

    inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext::getDeviceCellsPerTrackletTable()
    {
      return mGPUContext.getCellsPerTrackletTable();
    }

    inline std::array<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getTempTableArray()
    {
      return mTempTableArray;
    }

    inline std::array<CAGPUVector<CATracklet>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getTempTrackletArray()
    {
      return mTempTrackletArray;
    }

    inline std::array<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad - 1>& CAPrimaryVertexContext::getTempCellArray()
    {
      return mTempCellArray;
    }

    inline void CAPrimaryVertexContext::updateDeviceContext()
    {
      mGPUContextDevicePointer = CAGPUUniquePointer<CAGPUPrimaryVertexContext> { mGPUContext };
    }
#else
    inline std::array<std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
        CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext::getIndexTables()
    {
      return mIndexTables;
    }

    inline std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAPrimaryVertexContext::getTracklets()
    {
      return mTracklets;
    }

    inline std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>& CAPrimaryVertexContext::getTrackletsLookupTable()
    {
      return mTrackletsLookupTable;
    }
#endif

#endif /* TRACKINGITSU_INCLUDE_CAPRIMARYVERTEXCONTEXT_H_ */
