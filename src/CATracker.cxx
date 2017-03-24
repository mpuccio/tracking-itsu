/// \file CATracker.cxx
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

#include <cmath>

#include "CATracklet.h"

CATracker::CATracker(const CAEvent& event)
    : mEvent (event), mUsedClustersTable(event.getTotalClusters(), false), mLookupTables { }
{
  for (int iLayer = 0; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    mLookupTables[iLayer] = CALookupTable(event.getLayer(iLayer));
  }
}

int CATracker::clustersToTracks()
{

  for(int iIteration = 0; iIteration < CAConstants::ITS::TracksReconstructionIterations; ++iIteration) {

    makeCells(iIteration);

    //TODO: Implement
  }

  return 0;
}

void CATracker::makeCells(int iIteration)
{
  std::array<std::vector<CATracklet>, CAConstants::ITS::TrackletsPerRoad> tracklets;
  std::array<int, CAConstants::ITS::CellsPerRoad> trackletsLookupTable;

  for (int iLayer = 0; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const CALayer& currentLayer = mEvent.getLayer(iLayer);

    if (currentLayer.getClusters().empty()) {

      continue;
    }

    const CALayer& nextLayer = mEvent.getLayer(iLayer + 1);

    for (int iCluster = 0; iCluster < currentLayer.getClustersSize(); ++iCluster) {

      const CACluster& currentCluster = currentLayer.getCluster(iCluster);

      if (mUsedClustersTable[currentCluster.clusterId]) {

        continue;
      }

      const float tanLambda = (currentCluster.zCoordinate - mEvent.getPrimaryVertexZCoordinate())
          / currentCluster.rCoordinate;
      const float extz = tanLambda * (CAConstants::ITS::LayersRCoordinate[iLayer + 1] - currentCluster.rCoordinate)
          + currentCluster.zCoordinate;

      const std::vector<int> nextLayerClusters = mLookupTables[iLayer + 1].selectClusters(
          extz - 2 * CAConstants::LookupTable::ZCoordinateCut, extz + 2 * CAConstants::LookupTable::ZCoordinateCut,
          currentCluster.phiCoordinate - CAConstants::LookupTable::PhiCoordinateCut[iIteration],
          currentCluster.phiCoordinate + CAConstants::LookupTable::PhiCoordinateCut[iIteration]);

      const int nextLayerClustersNum = nextLayerClusters.size();

      for (int iNextLayerCluster = 0; iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

        const CACluster& nextCluster = nextLayer.getCluster(nextLayerClusters[iNextLayerCluster]);

        if (mUsedClustersTable[nextCluster.clusterId]) {

          continue;
        }

        const float deltaZ = std::abs(tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) +
            currentCluster.zCoordinate - nextCluster.zCoordinate);
        const float deltaPhi = std::abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate);


        if (deltaZ < CAConstants::ITS::TrackletMaxDeltaZThreshold[iLayer] && (
          deltaPhi < CAConstants::LookupTable::PhiCoordinateCut[iIteration] ||
          std::abs(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::LookupTable::PhiCoordinateCut[iIteration])) {

          const float doubletTanLambda = (currentCluster.zCoordinate - nextCluster.zCoordinate) /
              (currentCluster.rCoordinate - nextCluster.rCoordinate);
          const float doubletPhi = std::atan2(currentCluster.yCoordinate - nextCluster.yCoordinate,
              currentCluster.xCoordinate - nextCluster.xCoordinate);

          tracklets[iLayer].emplace_back(iCluster, iNextLayerCluster, doubletTanLambda, doubletPhi);
        }
      }
    }
  }

  for (int iCell = 0; iCell < CAConstants::ITS::CellsPerRoad; ++iCell) {

    if (tracklets[iCell + 1].empty() || tracklets[iCell + 1].empty()) {

      continue;
    }

    const int currentLayerTrackletsNum = tracklets[iCell].size();

    for(int iTracklet = 0; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const int nextLayerClusterIndex = tracklets[iCell][iTracklet].secondClusterIndex;

    }
  }
}
