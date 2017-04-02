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

#include <iostream>
#include <iomanip>
#include <functional>

#include <cmath>

#include "CAMathUtils.h"

namespace {

void evaluateTask(void (CATracker::*task)(CATrackerContext&), CATracker* tracker, const char *taskName,
    CATrackerContext& trackerContext)
{

  clock_t t1, t2;
  float diff;

  t1 = clock();

  (tracker->*task)(trackerContext);

  t2 = clock();
  diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
  std::cout << std::setw(2) << " - " << taskName << " completed in: " << diff << "ms" << std::endl;
}
}

CATracker::CATracker(const CAEvent& event)
    : mEvent(event), mUsedClustersTable(event.getTotalClusters(), CAConstants::ITS::UnusedIndex), mIndexTables { }
{
  for (int iLayer = 0; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    mIndexTables[iLayer] = CAIndexTable(event.getLayer(iLayer + 1));
  }
}

void CATracker::clustersToTracks()
{
  CATrackerContext trackerContext { };

  computeTracklets(trackerContext);
  computeCells(trackerContext);
  findCellsNeighbours(trackerContext);
  findTracks(trackerContext);
  computeMontecarloLabels(trackerContext);
}

void CATracker::clustersToTracksVerbose()
{
  CATrackerContext trackerContext { };

  evaluateTask(&CATracker::computeTracklets, this, "Tracklets Finding", trackerContext);
  evaluateTask(&CATracker::computeCells, this, "Cells Finding", trackerContext);
  evaluateTask(&CATracker::findCellsNeighbours, this, "Neighbours Finding", trackerContext);
  evaluateTask(&CATracker::findTracks, this, "Tracks Finding", trackerContext);
  evaluateTask(&CATracker::computeMontecarloLabels, this, "Computing Montecarlo Labels", trackerContext);

  std::cout << "Found " << trackerContext.roads.size() << " roads" << std::endl;
}

void CATracker::computeTracklets(CATrackerContext& trackerContext)
{

  for (int iLayer = 0; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const CALayer& currentLayer = mEvent.getLayer(iLayer);
    const CALayer& nextLayer = mEvent.getLayer(iLayer + 1);

    if (currentLayer.getClusters().empty() || nextLayer.getClusters().empty()) {

      continue;
    }

    const int currentLayerClustersNum = currentLayer.getClustersSize();

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      trackerContext.trackletsLookupTable[iLayer].resize(nextLayer.getClustersSize(), CAConstants::ITS::UnusedIndex);
    }

    for (int iCluster = 0; iCluster < currentLayerClustersNum; ++iCluster) {

      const CACluster& currentCluster = currentLayer.getCluster(iCluster);

      if (mUsedClustersTable[currentCluster.clusterId] != CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const float tanLambda = (currentCluster.zCoordinate - mEvent.getPrimaryVertexZCoordinate())
          / currentCluster.rCoordinate;
      const float directionZIntersection = tanLambda
          * (CAConstants::ITS::LayersRCoordinate[iLayer + 1] - currentCluster.rCoordinate) + currentCluster.zCoordinate;

      const std::vector<int> nextLayerBinsSubset = mIndexTables[iLayer].selectBins(
          directionZIntersection - 2 * CAConstants::Thresholds::ZCoordinateCut,
          directionZIntersection + 2 * CAConstants::Thresholds::ZCoordinateCut,
          currentCluster.phiCoordinate - CAConstants::Thresholds::PhiCoordinateCut,
          currentCluster.phiCoordinate + CAConstants::Thresholds::PhiCoordinateCut);

      if (nextLayerBinsSubset.empty()) {

        continue;
      }

      bool isFirstTrackletFromCurrentCluster = true;

      const int lastClusterBin = nextLayerBinsSubset.size() - 1;

      for (int iClusterBin = 0; iClusterBin <= lastClusterBin; ++iClusterBin) {

        const int currentBinIndex = nextLayerBinsSubset[iClusterBin];
        const int currentBinFirstCluster = mIndexTables[iLayer].getBin(currentBinIndex);
        const int nextBinFirstClusterIndex = mIndexTables[iLayer].getBin(currentBinIndex + 1);

        for (int iNextLayerCluster = currentBinFirstCluster; iNextLayerCluster < nextBinFirstClusterIndex;
            ++iNextLayerCluster) {

          const CACluster& nextCluster = nextLayer.getCluster(iNextLayerCluster);

          if (mUsedClustersTable[nextCluster.clusterId] != CAConstants::ITS::UnusedIndex) {

            continue;
          }

          const float deltaZ = std::abs(
              tanLambda * (nextCluster.rCoordinate - currentCluster.rCoordinate) + currentCluster.zCoordinate
                  - nextCluster.zCoordinate);
          const float deltaPhi = std::abs(currentCluster.phiCoordinate - nextCluster.phiCoordinate);

          if (deltaZ < CAConstants::Thresholds::TrackletMaxDeltaZThreshold[iLayer]
              && (deltaPhi < CAConstants::Thresholds::PhiCoordinateCut
                  || std::abs(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::PhiCoordinateCut)) {

            if (iLayer > 0 && isFirstTrackletFromCurrentCluster) {

              trackerContext.trackletsLookupTable[iLayer - 1][iCluster] = trackerContext.tracklets[iLayer].size();
              isFirstTrackletFromCurrentCluster = false;
            }

            const float trackletTanLambda = (currentCluster.zCoordinate - nextCluster.zCoordinate)
                / (currentCluster.rCoordinate - nextCluster.rCoordinate);
            const float trackletPhi = std::atan2(currentCluster.yCoordinate - nextCluster.yCoordinate,
                currentCluster.xCoordinate - nextCluster.xCoordinate);

            trackerContext.tracklets[iLayer].emplace_back(iCluster, iNextLayerCluster, trackletTanLambda, trackletPhi);
          }
        }
      }
    }
  }
}

void CATracker::computeCells(CATrackerContext& trackerContext)
{
  for (int iLayer = 0; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    if (trackerContext.tracklets[iLayer + 1].empty() || trackerContext.tracklets[iLayer].empty()) {

      continue;
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad - 1) {

      trackerContext.cellsLookupTable[iLayer].resize(trackerContext.tracklets[iLayer + 1].size(),
          CAConstants::ITS::UnusedIndex);
    }

    const int currentLayerTrackletsNum = trackerContext.tracklets[iLayer].size();

    for (int iTracklet = 0; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const CATracklet& currentTracklet = trackerContext.tracklets[iLayer][iTracklet];
      const int nextLayerClusterIndex = currentTracklet.secondClusterIndex;
      const int nextLayerFirstTrackletIndex = trackerContext.trackletsLookupTable[iLayer][nextLayerClusterIndex];

      if (nextLayerFirstTrackletIndex == CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const CACluster& firstCellCluster = mEvent.getLayer(iLayer).getCluster(currentTracklet.firstClusterIndex);
      const CACluster& secondCellCluster = mEvent.getLayer(iLayer + 1).getCluster(currentTracklet.secondClusterIndex);

      const float firstCellClusterQuadraticRCoordinate = firstCellCluster.rCoordinate * firstCellCluster.rCoordinate;
      const float secondCellClusterQuadraticRCoordinate = secondCellCluster.rCoordinate * secondCellCluster.rCoordinate;

      const std::array<float, 3> firstDeltaVector { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
          secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
              - firstCellClusterQuadraticRCoordinate };

      bool isFirstCellForCurrentTracklet = true;
      const int nextLayerTrackletsNum = trackerContext.tracklets[iLayer + 1].size();

      for (int iNextLayerTracklet = nextLayerFirstTrackletIndex;
          iNextLayerTracklet < nextLayerTrackletsNum
              && trackerContext.tracklets[iLayer + 1][iNextLayerTracklet].firstClusterIndex == nextLayerClusterIndex;
          ++iNextLayerTracklet) {

        const CATracklet& nextTracklet = trackerContext.tracklets[iLayer + 1][iNextLayerTracklet];
        const float deltaTanLambda = std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda);
        const float deltaPhi = std::abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate);

        if (deltaTanLambda < CAConstants::Thresholds::CellMaxDeltaTanLambdaThreshold
            && (deltaPhi < CAConstants::Thresholds::CellMaxDeltaPhiThreshold
                || std::abs(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::CellMaxDeltaPhiThreshold)) {

          const float averageTanLambda = 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda);
          const float directionZIntersection = -averageTanLambda * firstCellCluster.rCoordinate
              + firstCellCluster.zCoordinate;
          const float deltaZ = std::abs(directionZIntersection - mEvent.getPrimaryVertexZCoordinate());

          if (deltaZ < CAConstants::Thresholds::CellMaxDeltaZThreshold[iLayer]) {

            const CACluster& thirdCellCluster = mEvent.getLayer(iLayer + 2).getCluster(nextTracklet.secondClusterIndex);

            const float thirdCellClusterQuadraticRCoordinate = thirdCellCluster.rCoordinate
                * thirdCellCluster.rCoordinate;

            const std::array<float, 3> secondDeltaVector { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
                thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate
                    - firstCellClusterQuadraticRCoordinate };

            std::array<float, 3> cellPlaneNormalVector { CAMathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

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
                - normalizedPlaneVector[2] * secondCellClusterQuadraticRCoordinate;

            const float normalizedPlaneVectorQuadraticZCoordinate = normalizedPlaneVector[2] * normalizedPlaneVector[2];

            const float cellTrajectoryRadius = std::sqrt(
                (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector[2])
                    / (4.0f * normalizedPlaneVectorQuadraticZCoordinate));

            const std::array<float, 2> circleCenter { -0.5f * normalizedPlaneVector[0] / normalizedPlaneVector[2], -0.5f
                * normalizedPlaneVector[1] / normalizedPlaneVector[2] };

            const float distanceOfClosestApproach = std::abs(
                cellTrajectoryRadius
                    - std::sqrt(circleCenter[0] * circleCenter[0] + circleCenter[1] * circleCenter[1]));

            if (distanceOfClosestApproach
                > CAConstants::Thresholds::CellMaxDistanceOfClosestApproachThreshold[iLayer]) {

              continue;
            }

            const float cellTrajectoryCurvature = 1.0f / cellTrajectoryRadius;

            if (isFirstCellForCurrentTracklet && iLayer > 0) {

              trackerContext.cellsLookupTable[iLayer - 1][iTracklet] = trackerContext.cells[iLayer].size();
              isFirstCellForCurrentTracklet = false;
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

      if (nextLayerFirstCellIndex == CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const int nextLayerCellsNum = trackerContext.cells[iLayer + 1].size();

      for (int iNextLayerCell = nextLayerFirstCellIndex;
          iNextLayerCell < nextLayerCellsNum
              && trackerContext.cells[iLayer + 1][iNextLayerCell].getFirstTrackletIndex() == nextLayerTrackletIndex;
          ++iNextLayerCell) {

        CACell& nextCell = trackerContext.cells[iLayer + 1][iNextLayerCell];
        const std::array<float, 3> currentCellNormalVector = currentCell.getNormalVectorCoordinates();
        const std::array<float, 3> nextCellNormalVector = nextCell.getNormalVectorCoordinates();

        const std::array<float, 3> normalVectorsDeltaVector = { currentCellNormalVector[0] - nextCellNormalVector[0],
            currentCellNormalVector[1] - nextCellNormalVector[1], currentCellNormalVector[2] - nextCellNormalVector[2] };

        const float deltaNormalVectorsModulus = (normalVectorsDeltaVector[0] * normalVectorsDeltaVector[0])
            + (normalVectorsDeltaVector[1] * normalVectorsDeltaVector[1])
            + (normalVectorsDeltaVector[2] * normalVectorsDeltaVector[2]);

        const float deltaCurvature = std::abs(currentCell.getCurvature() - nextCell.getCurvature());

        if (deltaNormalVectorsModulus < CAConstants::Thresholds::NeighbourCellMaxNormalVectorsDelta[iLayer]
            && deltaCurvature < CAConstants::Thresholds::NeighbourCellMaxCurvaturesDelta[iLayer]) {

          nextCell.combineCells(currentCell, iCell);
        }
      }
    }
  }
}

void CATracker::findTracks(CATrackerContext& trackerContext)
{
  for (int iLevel = CAConstants::ITS::CellsPerRoad; iLevel > CAConstants::Thresholds::TracksMinLength; --iLevel) {

    const int minimumLevel = iLevel - 1;

    for (int iPreviousLayer = CAConstants::ITS::CellsPerRoad - 1; iPreviousLayer >= minimumLevel; --iPreviousLayer) {

      const int levelCellsNum = trackerContext.cells[iPreviousLayer].size();

      for (int iCell = 0; iCell < levelCellsNum; ++iCell) {

        CACell& currentCell = trackerContext.cells[iPreviousLayer][iCell];

        if (currentCell.getLevel() != iLevel) {

          continue;
        }

        trackerContext.roads.emplace_back(iPreviousLayer, iCell);

        const int cellNeighboursNum = currentCell.getNumberOfNeighbours();

        for (int iNeighbourCell = 0; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          if (iNeighbourCell > 0) {

            trackerContext.roads.emplace_back(iPreviousLayer, iCell);
          }

          traverseCellsTree(trackerContext, currentCell.getNeighbourCellId(iNeighbourCell), iPreviousLayer - 1);
        }

        currentCell.setLevel(0);
      }
    }
  }
}

void CATracker::traverseCellsTree(CATrackerContext& trackerContext, const int currentCellId, const int currentLayerId)
{
  if (currentLayerId < 0) {

    return;
  }

  CACell& currentCell = trackerContext.cells[currentLayerId][currentCellId];

  trackerContext.roads.back().addCell(currentLayerId, currentCellId);

  const int cellNeighboursNum = currentCell.getNumberOfNeighbours();

  for (int iNeighbourCell = 0; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

    if (iNeighbourCell > 0) {

      trackerContext.roads.push_back(trackerContext.roads.back());

    }

    traverseCellsTree(trackerContext, currentCell.getNeighbourCellId(iNeighbourCell), currentLayerId - 1);
  }

  currentCell.setLevel(0);
}

void CATracker::computeMontecarloLabels(CATrackerContext& trackerContext)
{
  /// Moore’s Voting Algorithm

  int roadsNum = trackerContext.roads.size();

  for (int iRoad = 0; iRoad < roadsNum; ++iRoad) {

    CARoad& currentRoad = trackerContext.roads[iRoad];

    int maxOccurrencesValue = CAConstants::ITS::UnusedIndex;
    int count;

    bool isFakeRoad = false;
    bool isFirstRoadCell = true;

    for (int iCell = 0; iCell < CAConstants::ITS::CellsPerRoad; ++iCell) {

      const int currentCellIndex = currentRoad[iCell];

      if (currentCellIndex == CAConstants::ITS::UnusedIndex) {

        if (isFirstRoadCell) {

          continue;

        } else {

          break;
        }
      }

      const CACell& currentCell = trackerContext.cells[iCell][currentCellIndex];

      if (isFirstRoadCell) {

        maxOccurrencesValue = mEvent.getLayer(iCell).getCluster(currentCell.getFirstClusterIndex()).monteCarlo;
        count = 1;

        const int secondMonteCarlo =
            mEvent.getLayer(iCell + 1).getCluster(currentCell.getSecondClusterIndex()).monteCarlo;

        if (secondMonteCarlo == maxOccurrencesValue) {

          ++count;

        } else {

          maxOccurrencesValue = secondMonteCarlo;
          count = 1;
          isFakeRoad = true;
        }

        isFirstRoadCell = false;
      }

      const int currentMonteCarlo = mEvent.getLayer(iCell + 2).getCluster(currentCell.getThirdClusterIndex()).monteCarlo;

      if (currentMonteCarlo == maxOccurrencesValue) {

        ++count;

      } else {

        --count;
        isFakeRoad = true;
      }

      if (count == 0) {

        maxOccurrencesValue = currentMonteCarlo;
        count = 1;
      }
    }

    currentRoad.setLabel(maxOccurrencesValue);
    currentRoad.setFakeRoad(isFakeRoad);
  }
}
