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

CATracker::CATracker(const CAEvent& event)
    : mEvent { event }, mUsedClustersTable(event.getTotalClusters(), false), mLookupTables { }
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
  std::vector<int> trackletsLookUpTable[CAConstants::ITS::CellsPerRoad];

  for (int iLayer = 0; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const CALayer& currentLayer = mEvent.getLayer(iLayer);

    if (currentLayer.getClusters().empty()) {

      continue;
    }

    const CALayer& nextLayer = mEvent.getLayer(iLayer + 1);

    if ((iLayer + 1) < CAConstants::ITS::TrackletsPerRoad) { //TODO: capire perchè ultima volta no

      trackletsLookUpTable[iLayer].resize(nextLayer.getClustersSize(), -1);
    }

    // FIXME: Non ne capisco il significato: in piu' potrebbe dare errore nel caso iL == 0
    //if (trackletsLookUpTable[iLayer - 1].size() == 0u) continue;

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
          currentCluster.phiCoordinate - CAConstants::LookupTable::phiCoordinateCut[iIteration],
          currentCluster.phiCoordinate + CAConstants::LookupTable::phiCoordinateCut[iIteration]);

      const int nextLayerClustersNum = nextLayerClusters.size();

      for (int iNextLayerCluster = 0; iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {

        const CACluster& nextCluster = nextLayer.getCluster(nextLayerClusters[iNextLayerCluster]);

        if (mUsedClustersTable[nextCluster.clusterId]) {

          continue;
        }
      }
    }
  }
}
