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

#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "CACell.h"
#include "CAConstants.h"
#include "CAEvent.h"
#include "CALayer.h"
#include "CAMathUtils.h"
#include "CAPrimaryVertexContext.h"
#include "CATrackingUtils.h"
#include "CATracklet.h"

#if defined(TRACKINGITSU_GPU_MODE)
# include "CAGPUUtils.h"
# include "CAGPUTrackingAPI.h"
#endif

namespace {

void evaluateTask(void (CATracker::*task)(CAPrimaryVertexContext&), CATracker* tracker, const char *taskName,
    CAPrimaryVertexContext& primaryVertexContext)
{

  clock_t t1 { }, t2 { };
  float diff { };

  t1 = clock();

  (tracker->*task)(primaryVertexContext);

  t2 = clock();
  diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
  std::cout << std::setw(2) << " - " << taskName << " completed in: " << diff << "ms" << std::endl;
}
}

CATracker::CATracker(const CAEvent& event)
    : mEvent(event), mUsedClustersTable(event.getTotalClusters(), CAConstants::ITS::UnusedIndex)
{
  // Nothing to do
}

std::vector<std::vector<CARoad>> CATracker::clustersToTracks()
{
  const int verticesNum { mEvent.getPrimaryVerticesNum() };
  std::vector<std::vector<CARoad>> roads { };
  roads.reserve(verticesNum);

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    CAPrimaryVertexContext primaryVertexContext { mEvent, iVertex };

    computeTracklets(primaryVertexContext);
    computeCells(primaryVertexContext);
    findCellsNeighbours(primaryVertexContext);
    findTracks(primaryVertexContext);
    computeMontecarloLabels(primaryVertexContext);

    roads.emplace_back(primaryVertexContext.roads);
  }

  return roads;
}

std::vector<std::vector<CARoad>> CATracker::clustersToTracksVerbose()
{
  const int verticesNum { mEvent.getPrimaryVerticesNum() };
  std::vector<std::vector<CARoad>> roads { };
  roads.reserve(verticesNum);

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    clock_t t1 { }, t2 { };
    float diff { };

    t1 = clock();

    CAPrimaryVertexContext primaryVertexContext { mEvent, iVertex };

    evaluateTask(&CATracker::computeTracklets, this, "Tracklets Finding", primaryVertexContext);
    evaluateTask(&CATracker::computeCells, this, "Cells Finding", primaryVertexContext);
    evaluateTask(&CATracker::findCellsNeighbours, this, "Neighbours Finding", primaryVertexContext);
    evaluateTask(&CATracker::findTracks, this, "Tracks Finding", primaryVertexContext);
    evaluateTask(&CATracker::computeMontecarloLabels, this, "Computing Montecarlo Labels", primaryVertexContext);

    t2 = clock();
    diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
    std::cout << std::setw(2) << " - Vertex " << iVertex + 1 << " completed in: " << diff << "ms" << std::endl;

    std::cout << "Found " << primaryVertexContext.roads.size() << " roads for vertex " << iVertex + 1 << std::endl;

    roads.emplace_back(primaryVertexContext.roads);
  }

  return roads;
}

std::vector<std::vector<CARoad>> CATracker::clustersToTracksMemoryBenchmark(std::ofstream& memoryBenchmarkOutputStream)
{
  const int verticesNum { mEvent.getPrimaryVerticesNum() };
  std::vector<std::vector<CARoad>> roads { };
  roads.reserve(verticesNum);

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    CAPrimaryVertexContext primaryVertexContext { mEvent, iVertex };

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.clusters[iLayer].size() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.tracklets[iLayer].capacity() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    computeTracklets(primaryVertexContext);

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.tracklets[iLayer].size() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.cells[iLayer].capacity() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    computeCells(primaryVertexContext);

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.cells[iLayer].size() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    findCellsNeighbours(primaryVertexContext);
    findTracks(primaryVertexContext);
    computeMontecarloLabels(primaryVertexContext);

    roads.emplace_back(primaryVertexContext.roads);

    memoryBenchmarkOutputStream << primaryVertexContext.roads.size() << std::endl;
  }

  return roads;
}

void CATracker::computeTracklets(CAPrimaryVertexContext& primaryVertexContext)
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    const CALayer& currentLayer { mEvent.getLayer(iLayer) };
    const CALayer& nextLayer { mEvent.getLayer(iLayer + 1) };

    if (currentLayer.getClusters().empty() || nextLayer.getClusters().empty()) {

      continue;
    }

    const int currentLayerClustersNum { currentLayer.getClustersSize() };

    for (int iCluster { 0 }; iCluster < currentLayerClustersNum; ++iCluster) {

      const CACluster& currentCluster { primaryVertexContext.clusters[iLayer][iCluster] };

      if (mUsedClustersTable[currentCluster.clusterId] != CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const float tanLambda { (currentCluster.zCoordinate
          - mEvent.getPrimaryVertex(primaryVertexContext.primaryVertexIndex)[2]) / currentCluster.rCoordinate };
      const float directionZIntersection { tanLambda
          * (CAConstants::ITS::LayersRCoordinate[iLayer + 1] - currentCluster.rCoordinate) + currentCluster.zCoordinate };
      const float zRangeMin = directionZIntersection - 2 * CAConstants::Thresholds::ZCoordinateCut;
      const float phiRangeMin = currentCluster.phiCoordinate - CAConstants::Thresholds::PhiCoordinateCut;
      const float zRangeMax = directionZIntersection + 2 * CAConstants::Thresholds::ZCoordinateCut;
      const float phiRangeMax = currentCluster.phiCoordinate + CAConstants::Thresholds::PhiCoordinateCut;

      if (zRangeMax < -CAConstants::ITS::LayersZCoordinate[iLayer + 1]
          || zRangeMin > CAConstants::ITS::LayersZCoordinate[iLayer + 1] || zRangeMin > zRangeMax) {

        continue;
      }

      const std::array<int, 4> selectedBinsRect { primaryVertexContext.indexTables[iLayer].getSelectedBinsRect(
          zRangeMin, phiRangeMin, zRangeMax, phiRangeMax) };
      const std::vector<std::pair<int, int>> nextLayerClustersSubset {
          primaryVertexContext.indexTables[iLayer].selectClusters(selectedBinsRect) };

#if defined(TRACKINGITSU_GPU_MODE)
      CAGPUTrackingAPI::getTrackletsFromCluster(primaryVertexContext, iLayer, iCluster, tanLambda,
          directionZIntersection, selectedBinsRect, nextLayerClustersSubset);
#else

      const int rowsNum { static_cast<int>(nextLayerClustersSubset.size()) };

      for (int iRow { 0 }; iRow < rowsNum; ++iRow) {

        const int firstRowClusterIndex { nextLayerClustersSubset[iRow].first };
        const int lastRowClusterIndex { firstRowClusterIndex + nextLayerClustersSubset[iRow].second - 1 };

        for (int iNextLayerCluster { firstRowClusterIndex }; iNextLayerCluster <= lastRowClusterIndex;
            ++iNextLayerCluster) {

          const CACluster& nextCluster { primaryVertexContext.clusters[iLayer + 1][iNextLayerCluster] };

          if (CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

            if (iLayer > 0
                && primaryVertexContext.trackletsLookupTable[iLayer - 1][iCluster] == CAConstants::ITS::UnusedIndex) {

              primaryVertexContext.trackletsLookupTable[iLayer - 1][iCluster] =
                  primaryVertexContext.tracklets[iLayer].size();
            }

            primaryVertexContext.tracklets[iLayer].emplace_back(iCluster, iNextLayerCluster, currentCluster,
                nextCluster);
          }
        }
      }
#endif
    }
  }

#if defined(TRACKINGITSU_GPU_MODE)
  for (int iLayer {0}; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    std::unique_ptr<int, void (*)(void*)> sizeUniquePointer =
    primaryVertexContext.dTracklets[iLayer].getSizeFromDevice();
    primaryVertexContext.dTracklets[iLayer].copyIntoVector(primaryVertexContext.tracklets[iLayer], *sizeUniquePointer);

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      primaryVertexContext.dTrackletsLookupTable[iLayer].copyIntoVector(
          primaryVertexContext.trackletsLookupTable[iLayer], mEvent.getLayer(iLayer + 1).getClustersSize());
    }
  }
#endif
}

void CATracker::computeCells(CAPrimaryVertexContext& primaryVertexContext)
{
  const std::array<float, 3>& primaryVertex { mEvent.getPrimaryVertex(primaryVertexContext.primaryVertexIndex) };

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    if (primaryVertexContext.tracklets[iLayer + 1].empty() || primaryVertexContext.tracklets[iLayer].empty()) {

      continue;
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad - 1) {

      primaryVertexContext.cellsLookupTable[iLayer].resize(primaryVertexContext.tracklets[iLayer + 1].size(),
          CAConstants::ITS::UnusedIndex);
    }

    const int currentLayerTrackletsNum { static_cast<int>(primaryVertexContext.tracklets[iLayer].size()) };

    for (int iTracklet { 0 }; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

      const CATracklet& currentTracklet { primaryVertexContext.tracklets[iLayer][iTracklet] };
      const int nextLayerClusterIndex { currentTracklet.secondClusterIndex };
      const int nextLayerFirstTrackletIndex { primaryVertexContext.trackletsLookupTable[iLayer][nextLayerClusterIndex] };

      if (nextLayerFirstTrackletIndex == CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const CACluster& firstCellCluster { primaryVertexContext.clusters[iLayer][currentTracklet.firstClusterIndex] };
      const CACluster& secondCellCluster { primaryVertexContext.clusters[iLayer + 1][currentTracklet.secondClusterIndex] };
      const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
      const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
      const std::array<float, 3> firstDeltaVector { { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
          secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
              - firstCellClusterQuadraticRCoordinate } };
      const int nextLayerTrackletsNum { static_cast<int>(primaryVertexContext.tracklets[iLayer + 1].size()) };

      for (int iNextLayerTracklet { nextLayerFirstTrackletIndex };
          iNextLayerTracklet < nextLayerTrackletsNum
              && primaryVertexContext.tracklets[iLayer + 1][iNextLayerTracklet].firstClusterIndex
                  == nextLayerClusterIndex; ++iNextLayerTracklet) {

        const CATracklet& nextTracklet { primaryVertexContext.tracklets[iLayer + 1][iNextLayerTracklet] };
        const float deltaTanLambda { std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda) };
        const float deltaPhi { std::abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate) };

        if (deltaTanLambda < CAConstants::Thresholds::CellMaxDeltaTanLambdaThreshold
            && (deltaPhi < CAConstants::Thresholds::CellMaxDeltaPhiThreshold
                || std::abs(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::CellMaxDeltaPhiThreshold)) {

          const float averageTanLambda { 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
          const float directionZIntersection { -averageTanLambda * firstCellCluster.rCoordinate
              + firstCellCluster.zCoordinate };
          const float deltaZ { std::abs(directionZIntersection - primaryVertex[2]) };

          if (deltaZ < CAConstants::Thresholds::CellMaxDeltaZThreshold[iLayer]) {

            const CACluster& thirdCellCluster {
                primaryVertexContext.clusters[iLayer + 2][nextTracklet.secondClusterIndex] };

            const float thirdCellClusterQuadraticRCoordinate { thirdCellCluster.rCoordinate
                * thirdCellCluster.rCoordinate };

            const std::array<float, 3> secondDeltaVector { { thirdCellCluster.xCoordinate
                - firstCellCluster.xCoordinate, thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate,
                thirdCellClusterQuadraticRCoordinate - firstCellClusterQuadraticRCoordinate } };

            std::array<float, 3> cellPlaneNormalVector { CAMathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

            const float vectorNorm { std::sqrt(
                cellPlaneNormalVector[0] * cellPlaneNormalVector[0]
                    + cellPlaneNormalVector[1] * cellPlaneNormalVector[1]
                    + cellPlaneNormalVector[2] * cellPlaneNormalVector[2]) };

            if (vectorNorm < CAConstants::Math::FloatMinThreshold
                || std::abs(cellPlaneNormalVector[2]) < CAConstants::Math::FloatMinThreshold) {

              continue;
            }

            const float inverseVectorNorm { 1.0f / vectorNorm };
            const std::array<float, 3> normalizedPlaneVector { { cellPlaneNormalVector[0] * inverseVectorNorm,
                cellPlaneNormalVector[1] * inverseVectorNorm, cellPlaneNormalVector[2] * inverseVectorNorm } };
            const float planeDistance { -normalizedPlaneVector[0] * (secondCellCluster.xCoordinate - primaryVertex[0])
                - (normalizedPlaneVector[1] * secondCellCluster.yCoordinate - primaryVertex[1])
                - normalizedPlaneVector[2] * secondCellClusterQuadraticRCoordinate };
            const float normalizedPlaneVectorQuadraticZCoordinate { normalizedPlaneVector[2] * normalizedPlaneVector[2] };
            const float cellTrajectoryRadius { std::sqrt(
                (1.0f - normalizedPlaneVectorQuadraticZCoordinate - 4.0f * planeDistance * normalizedPlaneVector[2])
                    / (4.0f * normalizedPlaneVectorQuadraticZCoordinate)) };
            const std::array<float, 2> circleCenter { { -0.5f * normalizedPlaneVector[0] / normalizedPlaneVector[2],
                -0.5f * normalizedPlaneVector[1] / normalizedPlaneVector[2] } };
            const float distanceOfClosestApproach { std::abs(
                cellTrajectoryRadius - std::sqrt(circleCenter[0] * circleCenter[0] + circleCenter[1] * circleCenter[1])) };

            if (distanceOfClosestApproach
                > CAConstants::Thresholds::CellMaxDistanceOfClosestApproachThreshold[iLayer]) {

              continue;
            }

            const float cellTrajectoryCurvature { 1.0f / cellTrajectoryRadius };

            if (iLayer > 0
                && primaryVertexContext.cellsLookupTable[iLayer - 1][iTracklet] == CAConstants::ITS::UnusedIndex) {

              primaryVertexContext.cellsLookupTable[iLayer - 1][iTracklet] = primaryVertexContext.cells[iLayer].size();
            }

            primaryVertexContext.cells[iLayer].emplace_back(currentTracklet.firstClusterIndex,
                nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex, iTracklet, iNextLayerTracklet,
                normalizedPlaneVector, cellTrajectoryCurvature);
          }
        }
      }
    }
  }
}

void CATracker::findCellsNeighbours(CAPrimaryVertexContext& primaryVertexContext)
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad - 1; ++iLayer) {

    if (primaryVertexContext.cells[iLayer + 1].empty() || primaryVertexContext.cellsLookupTable[iLayer].empty()) {

      continue;
    }

    int layerCellsNum { static_cast<int>(primaryVertexContext.cells[iLayer].size()) };

    for (int iCell { 0 }; iCell < layerCellsNum; ++iCell) {

      const CACell& currentCell { primaryVertexContext.cells[iLayer][iCell] };
      const int nextLayerTrackletIndex { currentCell.getSecondTrackletIndex() };
      const int nextLayerFirstCellIndex { primaryVertexContext.cellsLookupTable[iLayer][nextLayerTrackletIndex] };

      if (nextLayerFirstCellIndex == CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const int nextLayerCellsNum { static_cast<int>(primaryVertexContext.cells[iLayer + 1].size()) };

      for (int iNextLayerCell { nextLayerFirstCellIndex };
          iNextLayerCell < nextLayerCellsNum
              && primaryVertexContext.cells[iLayer + 1][iNextLayerCell].getFirstTrackletIndex()
                  == nextLayerTrackletIndex; ++iNextLayerCell) {

        CACell& nextCell { primaryVertexContext.cells[iLayer + 1][iNextLayerCell] };
        const std::array<float, 3> currentCellNormalVector { currentCell.getNormalVectorCoordinates() };
        const std::array<float, 3> nextCellNormalVector { nextCell.getNormalVectorCoordinates() };
        const std::array<float, 3> normalVectorsDeltaVector { { currentCellNormalVector[0] - nextCellNormalVector[0],
            currentCellNormalVector[1] - nextCellNormalVector[1], currentCellNormalVector[2] - nextCellNormalVector[2] } };

        const float deltaNormalVectorsModulus { (normalVectorsDeltaVector[0] * normalVectorsDeltaVector[0])
            + (normalVectorsDeltaVector[1] * normalVectorsDeltaVector[1])
            + (normalVectorsDeltaVector[2] * normalVectorsDeltaVector[2]) };
        const float deltaCurvature { std::abs(currentCell.getCurvature() - nextCell.getCurvature()) };

        if (deltaNormalVectorsModulus < CAConstants::Thresholds::NeighbourCellMaxNormalVectorsDelta[iLayer]
            && deltaCurvature < CAConstants::Thresholds::NeighbourCellMaxCurvaturesDelta[iLayer]) {

          nextCell.combineCells(currentCell, iCell);
        }
      }
    }
  }
}

void CATracker::findTracks(CAPrimaryVertexContext& primaryVertexContext)
{
  for (int iLevel { CAConstants::ITS::CellsPerRoad }; iLevel >= CAConstants::Thresholds::CellsMinLevel; --iLevel) {

    const int minimumLevel { iLevel - 1 };

    for (int iLayer { CAConstants::ITS::CellsPerRoad - 1 }; iLayer >= minimumLevel; --iLayer) {

      const int levelCellsNum { static_cast<int>(primaryVertexContext.cells[iLayer].size()) };

      for (int iCell { 0 }; iCell < levelCellsNum; ++iCell) {

        CACell& currentCell { primaryVertexContext.cells[iLayer][iCell] };

        if (currentCell.getLevel() != iLevel) {

          continue;
        }

        primaryVertexContext.roads.emplace_back(iLayer, iCell);

        const int cellNeighboursNum { currentCell.getNumberOfNeighbours() };

        for (int iNeighbourCell { 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          if (iNeighbourCell > 0) {

            primaryVertexContext.roads.emplace_back(iLayer, iCell);
          }

          traverseCellsTree(primaryVertexContext, currentCell.getNeighbourCellId(iNeighbourCell), iLayer - 1);
        }

        currentCell.setLevel(0);
      }
    }
  }
}

void CATracker::traverseCellsTree(CAPrimaryVertexContext& primaryVertexContext, const int currentCellId,
    const int currentLayerId)
{
  if (currentLayerId < 0) {

    return;
  }

  CACell& currentCell { primaryVertexContext.cells[currentLayerId][currentCellId] };

  primaryVertexContext.roads.back().addCell(currentLayerId, currentCellId);

  const int cellNeighboursNum { currentCell.getNumberOfNeighbours() };

  for (int iNeighbourCell { 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

    if (iNeighbourCell > 0) {

      primaryVertexContext.roads.push_back(primaryVertexContext.roads.back());

    }

    traverseCellsTree(primaryVertexContext, currentCell.getNeighbourCellId(iNeighbourCell), currentLayerId - 1);
  }

  currentCell.setLevel(0);
}

void CATracker::computeMontecarloLabels(CAPrimaryVertexContext& primaryVertexContext)
{
  /// Moore’s Voting Algorithm

  int roadsNum { static_cast<int>(primaryVertexContext.roads.size()) };

  for (int iRoad { 0 }; iRoad < roadsNum; ++iRoad) {

    CARoad& currentRoad { primaryVertexContext.roads[iRoad] };
    int maxOccurrencesValue { CAConstants::ITS::UnusedIndex };
    int count { 0 };
    bool isFakeRoad { false };
    bool isFirstRoadCell { true };

    for (int iCell { 0 }; iCell < CAConstants::ITS::CellsPerRoad; ++iCell) {

      const int currentCellIndex { currentRoad[iCell] };

      if (currentCellIndex == CAConstants::ITS::UnusedIndex) {

        if (isFirstRoadCell) {

          continue;

        } else {

          break;
        }
      }

      const CACell& currentCell { primaryVertexContext.cells[iCell][currentCellIndex] };

      if (isFirstRoadCell) {

        maxOccurrencesValue = primaryVertexContext.clusters[iCell][currentCell.getFirstClusterIndex()].monteCarloId;
        count = 1;

        const int secondMonteCarlo {
            primaryVertexContext.clusters[iCell + 1][currentCell.getSecondClusterIndex()].monteCarloId };

        if (secondMonteCarlo == maxOccurrencesValue) {

          ++count;

        } else {

          maxOccurrencesValue = secondMonteCarlo;
          count = 1;
          isFakeRoad = true;
        }

        isFirstRoadCell = false;
      }

      const int currentMonteCarlo {
          primaryVertexContext.clusters[iCell + 2][currentCell.getThirdClusterIndex()].monteCarloId };

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
