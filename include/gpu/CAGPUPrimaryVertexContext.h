/// \file CAGPUPrimaryVertexContext.h
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

#include "CACluster.h"
#include "CAConstants.h"
#include "CADefinitions.h"
#include "CAGPUArray.h"
#include "CAGPUUniquePointer.h"
#include "CAGPUVector.h"
#include "CAPrimaryVertexContext.h"
#include "CATracklet.h"

using namespace TRACKINGITSU_TARGET_NAMESPACE;

class CAGPUPrimaryVertexContext
  final
  {
    public:
      CAGPUPrimaryVertexContext(const float3&,
          const std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>&,
          const std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>&,
          const std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>&);

      GPU_DEVICE const float3& getPrimaryVertex();
      GPU_HOST_DEVICE CAGPUArray<CAGPUVector<CACluster>,
          CAConstants::ITS::LayersNumber>& getClusters();
      GPU_DEVICE CAGPUArray<CAGPUArray<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
          CAConstants::ITS::TrackletsPerRoad>& getIndexTables();
      GPU_HOST_DEVICE CAGPUArray<CAGPUVector<CATracklet>,
          CAConstants::ITS::TrackletsPerRoad>& getTracklets();
      GPU_HOST_DEVICE CAGPUArray<CAGPUVector<int>,
          CAConstants::ITS::CellsPerRoad>& getTrackletsLookupTable();
      GPU_HOST_DEVICE CAGPUArray<CAGPUVector<int>,
          CAConstants::ITS::CellsPerRoad>& getTrackletsPerClusterTable();
      GPU_HOST_DEVICE CAGPUArray<CAGPUVector<CACell>,
          CAConstants::ITS::CellsPerRoad>& getCells();
      GPU_HOST_DEVICE CAGPUArray<CAGPUVector<int>,
          CAConstants::ITS::CellsPerRoad - 1>& getCellsLookupTable();

    private:
      const CAGPUUniquePointer<float3> mPrimaryVertex;
      CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber> mClusters;
      CAGPUArray<CAGPUArray<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
          CAConstants::ITS::TrackletsPerRoad> mIndexTables;
      CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> mTracklets;
      CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad> mTrackletsLookupTable;
      CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad> mTrackletsPerClusterTable;
      CAGPUArray<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad> mCells;
      CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1> mCellsLookupTable;
  };

  GPU_DEVICE inline const float3& CAGPUPrimaryVertexContext::getPrimaryVertex()
  {
    return *mPrimaryVertex;
  }

  GPU_HOST_DEVICE inline CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber>& CAGPUPrimaryVertexContext::getClusters()
  {
    return mClusters;
  }

  GPU_DEVICE inline CAGPUArray<CAGPUArray<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>,
      CAConstants::ITS::TrackletsPerRoad>& CAGPUPrimaryVertexContext::getIndexTables()
  {
    return mIndexTables;
  }

  GPU_DEVICE inline CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& CAGPUPrimaryVertexContext::getTracklets()
  {
    return mTracklets;
  }

  GPU_DEVICE inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAGPUPrimaryVertexContext::getTrackletsLookupTable()
  {
    return mTrackletsLookupTable;
  }

  GPU_DEVICE inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& CAGPUPrimaryVertexContext::getTrackletsPerClusterTable()
  {
    return mTrackletsPerClusterTable;
  }

  GPU_HOST_DEVICE inline CAGPUArray<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad>& CAGPUPrimaryVertexContext::getCells()
  {
    return mCells;
  }

  GPU_HOST_DEVICE inline CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& CAGPUPrimaryVertexContext::getCellsLookupTable()
  {
    return mCellsLookupTable;
  }

  template<>
  class CAPrimaryVertexContext<true>
  {
    public:
      explicit CAPrimaryVertexContext(const CAEvent&, const int);

      CAPrimaryVertexContext(const CAPrimaryVertexContext&) = delete;
      CAPrimaryVertexContext &operator=(const CAPrimaryVertexContext&) = delete;

      const float3& getPrimaryVertex();
      std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber>& getClusters();
      std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& getTracklets();
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad>& getTrackletsLookupTable();
      std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad>& getCells();
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
      std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1>& getCellsNeighbours();
      std::vector<CARoad>& getRoads();

      CAGPUPrimaryVertexContext& getDeviceContext();
      CAGPUArray<CAGPUVector<CACluster>, CAConstants::ITS::LayersNumber>& getDeviceClusters();
      CAGPUArray<CAGPUVector<CATracklet>, CAConstants::ITS::TrackletsPerRoad>& getDeviceTracklets();
      CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& getDeviceTrackletsLookupTable();
      CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad>& getDeviceTrackletsPerClustersTable();
      CAGPUArray<CAGPUVector<CACell>, CAConstants::ITS::CellsPerRoad>& getDeviceCells();
      CAGPUArray<CAGPUVector<int>, CAConstants::ITS::CellsPerRoad - 1>& getDeviceCellsLookupTable();
      void updateDeviceContext();

    private:
      const float3 mPrimaryVertex;
      std::array<std::vector<CACluster>, CAConstants::ITS::LayersNumber> mClusters;
      std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> mTracklets;
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad> mTrackletsLookupTable;
      std::array<std::vector<CACell>, CAConstants::ITS::CellsPerRoad> mCells;
      std::array<std::vector<int>, CAConstants::ITS::CellsPerRoad - 1> mCellsLookupTable;
      std::array<std::vector<std::vector<int>>, CAConstants::ITS::CellsPerRoad - 1> mCellsNeighbours;
      std::vector<CARoad> mRoads;

      CAGPUPrimaryVertexContext mGPUContext;
      CAGPUUniquePointer<CAGPUPrimaryVertexContext> mGPUContextDevicePointer;
  };
