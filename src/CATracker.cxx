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
#include <iostream>

#include "CAMathUtils.h"

namespace {

constexpr int UnusedIndex = -1;
}

CATracker::CATracker(const CAEvent& event)
    : mEvent(event), mUsedClustersTable(event.getTotalClusters(), false), mIndexTables { }
{
  for (int iLayer = 0; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    mIndexTables[iLayer] = CAIndexTable(event.getLayer(iLayer));
  }
}

int CATracker::clustersToTracks()
{

  for (int iIteration = 0; iIteration < CAConstants::ITS::TracksReconstructionIterations; ++iIteration) {

    CATrackerContext trackerContext{iIteration};

    computeTracklets(trackerContext);
    computeCells(trackerContext);
    findCellsNeighbours(trackerContext);
    findTracks(trackerContext);
  }

  return 0;
}

void CATracker::computeTracklets(CATrackerContext& trackerContext)
{

  for (int iLayer = 0; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const CALayer& currentLayer = mEvent.getLayer(iLayer);

    if (currentLayer.getClusters().empty()) {

      continue;
    }

    const CALayer& nextLayer = mEvent.getLayer(iLayer + 1);

    const int currentLayerClustersNum = currentLayer.getClustersSize();

    if (iLayer > 0) {

      trackerContext.trackletsLookupTable[iLayer - 1].resize(nextLayer.getClustersSize(), UnusedIndex);
    }

    for (int iCluster = 0; iCluster < currentLayerClustersNum; ++iCluster) {

      const CACluster& currentCluster = currentLayer.getCluster(iCluster);

      if (mUsedClustersTable[currentCluster.clusterId]) {

        continue;
      }

      const float tanLambda = (currentCluster.zCoordinate - mEvent.getPrimaryVertexZCoordinate())
          / currentCluster.rCoordinate;
      const float directionZIntersection = tanLambda
          * (CAConstants::Thresholds::LayersRCoordinate[iLayer + 1] - currentCluster.rCoordinate)
          + currentCluster.zCoordinate;

      const std::vector<int> nextLayerClustersSubset = mIndexTables[iLayer + 1].selectClusters(
          directionZIntersection - 2 * CAConstants::Thresholds::ZCoordinateCut,
          directionZIntersection + 2 * CAConstants::Thresholds::ZCoordinateCut,
          currentCluster.phiCoordinate - CAConstants::Thresholds::PhiCoordinateCut[trackerContext.iteration],
          currentCluster.phiCoordinate + CAConstants::Thresholds::PhiCoordinateCut[trackerContext.iteration]);

      bool isFirstTrackletFromCurrentCluster = true;

      if (nextLayerClustersSubset.empty()) {

        continue;
      }

      const int nextLayerClusterSubsetNum = nextLayerClustersSubset.size();

      for (int iNextLayerCluster = 0; iNextLayerCluster < nextLayerClusterSubsetNum; ++iNextLayerCluster) {

        const CACluster& nextCluster = nextLayer.getCluster(nextLayerClustersSubset[iNextLayerCluster]);

        if (mUsedClustersTable[nextCluster.clusterId]) {

          continue;
        }

        const float deltaZ = std::abs(
            tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate
                - nextCluster.zCoordinate);
        const float deltaPhi = std::abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate);

        if (deltaZ < CAConstants::Thresholds::TrackletMaxDeltaZThreshold[trackerContext.iteration][iLayer]
            && (deltaPhi < CAConstants::Thresholds::PhiCoordinateCut[trackerContext.iteration]
                || std::abs(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::PhiCoordinateCut[trackerContext.iteration])) {

          if (iLayer > 0 && isFirstTrackletFromCurrentCluster) {

            trackerContext.trackletsLookupTable[iLayer - 1][iCluster] = trackerContext.tracklets[iLayer].size();
            isFirstTrackletFromCurrentCluster = false;
          }

          const float doubletTanLambda = (currentCluster.zCoordinate - nextCluster.zCoordinate)
              / (currentCluster.rCoordinate - nextCluster.rCoordinate);
          const float doubletPhi = std::atan2(currentCluster.yCoordinate - nextCluster.yCoordinate,
              currentCluster.xCoordinate - nextCluster.xCoordinate);

          trackerContext.tracklets[iLayer].emplace_back(iCluster, iNextLayerCluster, doubletTanLambda, doubletPhi);
        }
      }
    }
  }
}

void CATracker::computeCells(CATrackerContext& trackerContext)
{

  for (int iLayer = 0; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    if (trackerContext.tracklets[iLayer + 1].empty() || trackerContext.tracklets[iLayer + 1].empty()) {

      continue;
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      trackerContext.cellsLookupTable[iLayer].resize(trackerContext.tracklets[iLayer + 1].size(), UnusedIndex);
    }

    const int currentLayerTrackletsNum = trackerContext.tracklets[iLayer].size();

    for (int iTracklet = 0; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      bool isFirstTrackletCell = true;

      const CATracklet& currentTracklet = trackerContext.tracklets[iLayer][iTracklet];
      const int nextLayerClusterIndex = currentTracklet.secondClusterIndex;
      const int nextLayerFirstTrackletIndex = trackerContext.trackletsLookupTable[iLayer][nextLayerClusterIndex];

      if (nextLayerClusterIndex == UnusedIndex) {

        continue;
      }

      const int nextLayerTrackletsNum = trackerContext.tracklets[iLayer + 1].size();

      for (int iNextLayerTracklet = nextLayerFirstTrackletIndex;
          iNextLayerTracklet < nextLayerTrackletsNum
              && trackerContext.tracklets[iLayer + 1][iNextLayerTracklet].firstClusterIndex != nextLayerClusterIndex;
          ++iNextLayerTracklet) {

        const CATracklet& nextTracklet = trackerContext.tracklets[iLayer + 1][iNextLayerTracklet];
        const float deltaTanLambda = std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda);
        const float deltaPhi = std::abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate);

        if (deltaTanLambda < CAConstants::Thresholds::CellMaxDeltaTanLambdaThreshold[trackerContext.iteration]
            && (deltaPhi < CAConstants::Thresholds::CellMaxDeltaPhiThreshold[trackerContext.iteration]
                || std::abs(deltaPhi - CAConstants::Math::TwoPi)
                    < CAConstants::Thresholds::CellMaxDeltaPhiThreshold[trackerContext.iteration])) {

          const float averageTanLambda = 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda);
          const CACluster& firstCellCluster = mEvent.getLayer(iLayer).getCluster(currentTracklet.firstClusterIndex);
          const float directionZIntersection = -averageTanLambda * firstCellCluster.rCoordinate
              + firstCellCluster.zCoordinate;
          const float deltaZ = std::abs(directionZIntersection - mEvent.getPrimaryVertexZCoordinate());

          if (deltaZ < CAConstants::Thresholds::CellMaxDeltaZThreshold[trackerContext.iteration][iLayer]) {

            const CACluster& secondCellCluster = mEvent.getLayer(iLayer + 1).getCluster(nextTracklet.firstClusterIndex);
            const CACluster& thirdCellCluster = mEvent.getLayer(iLayer + 2).getCluster(nextTracklet.secondClusterIndex);

            const std::array<float, 3> firstVector { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellCluster.zCoordinate
                    - firstCellCluster.zCoordinate, };

            const std::array<float, 3> secondVector { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellCluster.zCoordinate
                    - firstCellCluster.zCoordinate, };

            std::array<float, 3> cellPlaneNormalVector { (firstVector[1] * secondVector[2])
                - (firstVector[2] * secondVector[1]), (firstVector[2] * secondVector[0])
                - (firstVector[0] * secondVector[2]), (firstVector[0] * secondVector[1])
                - (firstVector[1] * secondVector[0]) };

            const float vectorNorm = std::sqrt(
                cellPlaneNormalVector[0] * cellPlaneNormalVector[0]
                    + cellPlaneNormalVector[1] * cellPlaneNormalVector[1]
                    + cellPlaneNormalVector[2] * cellPlaneNormalVector[2]);

            if (vectorNorm < CAConstants::Math::FloatMinThreshold
                || std::abs(cellPlaneNormalVector[2]) < CAConstants::Math::FloatMinThreshold) {

              continue;
            }

            const float inverseVectorNorm = 1.0f / vectorNorm;

            const std::array<float, 3> normalizedPlaneVector { cellPlaneNormalVector[0] * inverseVectorNorm,
                cellPlaneNormalVector[1] * inverseVectorNorm, cellPlaneNormalVector[2] * inverseVectorNorm };

            const float planeDistance = -normalizedPlaneVector[0] * secondCellCluster.xCoordinate
                - normalizedPlaneVector[1] * secondCellCluster.yCoordinate
                - normalizedPlaneVector[2] * secondCellCluster.zCoordinate;

            const float cellTrajectoryRadius = std::sqrt(
                (1.0f - normalizedPlaneVector[2] * normalizedPlaneVector[2]
                    - 4.0f * planeDistance * normalizedPlaneVector[2])
                    / (4.0f * normalizedPlaneVector[2] * normalizedPlaneVector[2]));

            const std::array<float, 2> circleCenter { -0.5f * normalizedPlaneVector[0] / normalizedPlaneVector[2], -0.5f
                * normalizedPlaneVector[1] / normalizedPlaneVector[2] };

            const float distanceOfClosestApproach = std::abs(
                cellTrajectoryRadius
                    - std::sqrt(circleCenter[0] * circleCenter[0] + circleCenter[1] * circleCenter[1]));

            if (distanceOfClosestApproach
                > CAConstants::Thresholds::CellMaxDistanceOfClosestApproachThreshold[trackerContext.iteration][iLayer]) {

              continue;
            }

            const float cellTrajectoryCurvature = 1.0f / cellTrajectoryRadius;

            if (isFirstTrackletCell && iLayer > 0) {

              trackerContext.cellsLookupTable[iLayer - 1][iTracklet] = trackerContext.cells[iLayer].size();
              isFirstTrackletCell = false;
            }

            trackerContext.cells[iLayer].emplace_back(currentTracklet.firstClusterIndex, nextTracklet.firstClusterIndex,
                nextTracklet.secondClusterIndex, iTracklet, iNextLayerTracklet, normalizedPlaneVector,
                cellTrajectoryCurvature);
          }
        }
      }
    }
  }
}

void CATracker::findCellsNeighbours(CATrackerContext& trackerContext)
{
  for (int iLayer = 0; iLayer < CAConstants::ITS::CellsPerRoad - 1; ++iLayer) {

    if (trackerContext.cells[iLayer + 1].empty() || trackerContext.cellsLookupTable[iLayer].empty()) {

      continue;
    }

    int layerCellsNum = trackerContext.cells[iLayer].size();

    for (int iCell = 0; iCell < layerCellsNum; ++iCell) {

      const CACell& currentCell = trackerContext.cells[iLayer][iCell];

      const int nextLayerTrackletIndex = currentCell.getSecondTrackletIndex();
      const int nextLayerFirstCellIndex = trackerContext.cellsLookupTable[iLayer][nextLayerTrackletIndex];

      if (nextLayerFirstCellIndex == UnusedIndex) {

        continue;
      }

      const int nextLayerCellsNum = trackerContext.cells[iLayer + 1].size();

      for (int iNextLayerCell = nextLayerFirstCellIndex;
          iNextLayerCell < nextLayerCellsNum
              && trackerContext.cells[iLayer + 1][iNextLayerCell].getFirstTrackletIndex() != nextLayerTrackletIndex;
          ++iNextLayerCell) {

        CACell& nextCell = trackerContext.cells[iLayer + 1][iNextLayerCell];
        const std::array<float, 3> currentCellNormalVector = currentCell.getNormalVectorCoordinates();
        const std::array<float, 3> nextCellNormalVector = nextCell.getNormalVectorCoordinates();

        const float deltaNormalVectors = (currentCellNormalVector[0] - nextCellNormalVector[0])
            * (currentCellNormalVector[0] - nextCellNormalVector[0])
            + (currentCellNormalVector[1] - nextCellNormalVector[1])
                * (currentCellNormalVector[1] - nextCellNormalVector[1])
            + (currentCellNormalVector[2] - nextCellNormalVector[2])
                * (currentCellNormalVector[2] - nextCellNormalVector[2]);

        const float deltaCurvature = std::abs(currentCell.getCurvature() - nextCell.getCurvature());

        if (deltaNormalVectors < CAConstants::Thresholds::NeighbourCellMaxNormalVectorsDelta[trackerContext.iteration][iLayer]
            && deltaCurvature < CAConstants::Thresholds::NeighbourCellMaxCurvaturesDelta[trackerContext.iteration][iCell]) {

          nextCell.combineCells(currentCell, iCell);
        }
      }
    }
  }
}

void CATracker::findTracks(CATrackerContext& trackerContext)
{
  const int levelLimit = CAConstants::Thresholds::TracksMinLength[trackerContext.iteration];

  for(int iLayer = CAConstants::ITS::CellsPerRoad; iLayer > levelLimit; --iLayer) {

    for(int iPreviousLayer = CAConstants::ITS::CellsPerRoad - 1; iPreviousLayer > iLayer; --iPreviousLayer) {

      const int levelCellsNum = trackerContext.cells[iPreviousLayer].size();

      for(int iCell = 0; iCell < levelCellsNum; ++iCell) {

        CACell& currentCell = trackerContext.cells[iPreviousLayer][iCell];

        if(currentCell.getLevel() != iLayer) {

          continue;
        }

        trackerContext.roads.emplace_back(iPreviousLayer, iCell);

        const int cellNeighboursNum = currentCell.getNumberOfNeighbours();

        for(int iNeighbourCell = 0; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          if(iNeighbourCell > 0) {

            trackerContext.roads.emplace_back(iPreviousLayer, iCell);
          }

          traverseCellsTree(trackerContext, iPreviousLayer - 1, currentCell.getNeighbourCellId(iNeighbourCell));
        }

        currentCell.setLevel(0);
      }
    }
  }
}

void CATracker::traverseCellsTree(CATrackerContext& trackerContext, const int currentCellId, const int currentLayerId)
{
  if(currentLayerId < 0) {

    return;
  }

  CACell& currentCell = trackerContext.cells[currentLayerId][currentCellId];
  CARoad& currentRoad = trackerContext.roads.back();

  currentRoad.addCell(currentLayerId, currentCellId);

  const int cellNeighboursNum = currentCell.getNumberOfNeighbours();

  for(int iNeighbourCell = 0; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

    if(iNeighbourCell > 0) {

      trackerContext.roads.push_back(currentRoad);

    }

    traverseCellsTree(trackerContext, currentLayerId - 1, currentCell.getNeighbourCellId(iNeighbourCell));
  }

  currentCell.setLevel(0);
}
