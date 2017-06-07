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
#include "CADefinitions.h"
#include "CAEvent.h"
#include "CAIndexTableUtils.h"
#include "CALayer.h"
#include "CAMathUtils.h"
#include "CATracklet.h"
#include "CATrackingUtils.h"

#if TRACKINGITSU_GPU_MODE
# include "CAGPUPrimaryVertexContext.h"
#else
# include "CAPrimaryVertexContext.h"
#endif

template<>
void CATrackerTraits<false>::computeLayerTracklets(Context& primaryVertexContext, const int layerIndex,
    const std::array<float, 3> &primaryVertex)
{
  const int currentLayerClustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex].size()) };

  for (int iCluster { 0 }; iCluster < currentLayerClustersNum; ++iCluster) {

    const CACluster& currentCluster { primaryVertexContext.getClusters()[layerIndex][iCluster] };

    /*if (mUsedClustersTable[currentCluster.clusterId] != CAConstants::ITS::UnusedIndex) {

     continue;
     }*/

    const float tanLambda { (currentCluster.zCoordinate - primaryVertex[2]) / currentCluster.rCoordinate };
    const float directionZIntersection { tanLambda
        * (CAConstants::ITS::LayersRCoordinate()[layerIndex + 1] - currentCluster.rCoordinate)
        + currentCluster.zCoordinate };

    const GPU_ARRAY<int, 4> selectedBinsRect { CATrackingUtils::getBinsRect(currentCluster, layerIndex,
        directionZIntersection) };

    if (selectedBinsRect == CATrackingUtils::EmptyBinsRect) {

      continue;
    }

    int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

    if (phiBinsNum < 0) {

      phiBinsNum += CAConstants::IndexTable::PhiBins;
    }

    for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
        iPhiBin = ++iPhiBin == CAConstants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

      const int firstBinIndex { CAIndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };
      const int maxBinIndex { firstBinIndex + selectedBinsRect[2] - selectedBinsRect[0] + 1 };
      const int firstRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][firstBinIndex];
      const int maxRowClusterIndex = primaryVertexContext.getIndexTables()[layerIndex][maxBinIndex];

      for (int iNextLayerCluster { firstRowClusterIndex }; iNextLayerCluster <= maxRowClusterIndex;
          ++iNextLayerCluster) {

        const CACluster& nextCluster { primaryVertexContext.getClusters()[layerIndex + 1][iNextLayerCluster] };

        if (CATrackingUtils::isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

          if (layerIndex > 0
              && primaryVertexContext.getTrackletsLookupTable()[layerIndex - 1][iCluster]
                  == CAConstants::ITS::UnusedIndex) {

            primaryVertexContext.getTrackletsLookupTable()[layerIndex - 1][iCluster] =
                primaryVertexContext.getTracklets()[layerIndex].size();
          }

          primaryVertexContext.getTracklets()[layerIndex].emplace_back(iCluster, iNextLayerCluster, currentCluster,
              nextCluster);
        }
      }
    }
  }
}

template<>
void CATrackerTraits<false>::postProcessTracklets(Context& primaryVertexContext)
{
  // Nothing to do
}

template<>
void CATrackerTraits<false>::computeLayerCells(Context& primaryVertexContext, const int layerIndex,
    const std::array<float, 3> &primaryVertex)
{
  if (primaryVertexContext.getTracklets()[layerIndex + 1].empty()
      || primaryVertexContext.getTracklets()[layerIndex].empty()) {

    return;
  }

  const int currentLayerTrackletsNum { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex].size()) };

  for (int iTracklet { 0 }; iTracklet < currentLayerTrackletsNum; ++iTracklet) {

    const CATracklet& currentTracklet { primaryVertexContext.getTracklets()[layerIndex][iTracklet] };
    const int nextLayerClusterIndex { currentTracklet.secondClusterIndex };
    const int nextLayerFirstTrackletIndex {
        primaryVertexContext.getTrackletsLookupTable()[layerIndex][nextLayerClusterIndex] };

    if (nextLayerFirstTrackletIndex == CAConstants::ITS::UnusedIndex) {

      continue;
    }

    const CACluster& firstCellCluster {
        primaryVertexContext.getClusters()[layerIndex][currentTracklet.firstClusterIndex] };
    const CACluster& secondCellCluster {
        primaryVertexContext.getClusters()[layerIndex + 1][currentTracklet.secondClusterIndex] };
    const float firstCellClusterQuadraticRCoordinate { firstCellCluster.rCoordinate * firstCellCluster.rCoordinate };
    const float secondCellClusterQuadraticRCoordinate { secondCellCluster.rCoordinate * secondCellCluster.rCoordinate };
    const std::array<float, 3> firstDeltaVector { { secondCellCluster.xCoordinate - firstCellCluster.xCoordinate,
        secondCellCluster.yCoordinate - firstCellCluster.yCoordinate, secondCellClusterQuadraticRCoordinate
            - firstCellClusterQuadraticRCoordinate } };
    const int nextLayerTrackletsNum { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].size()) };

    for (int iNextLayerTracklet { nextLayerFirstTrackletIndex };
        iNextLayerTracklet < nextLayerTrackletsNum
            && primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet].firstClusterIndex
                == nextLayerClusterIndex; ++iNextLayerTracklet) {

      const CATracklet& nextTracklet { primaryVertexContext.getTracklets()[layerIndex + 1][iNextLayerTracklet] };
      const float deltaTanLambda { std::abs(currentTracklet.tanLambda - nextTracklet.tanLambda) };
      const float deltaPhi { std::abs(currentTracklet.phiCoordinate - nextTracklet.phiCoordinate) };

      if (deltaTanLambda < CAConstants::Thresholds::CellMaxDeltaTanLambdaThreshold
          && (deltaPhi < CAConstants::Thresholds::CellMaxDeltaPhiThreshold
              || std::abs(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::CellMaxDeltaPhiThreshold)) {

        const float averageTanLambda { 0.5f * (currentTracklet.tanLambda + nextTracklet.tanLambda) };
        const float directionZIntersection { -averageTanLambda * firstCellCluster.rCoordinate
            + firstCellCluster.zCoordinate };
        const float deltaZ { std::abs(directionZIntersection - primaryVertex[2]) };

        if (deltaZ < CAConstants::Thresholds::CellMaxDeltaZThreshold()[layerIndex]) {

          const CACluster& thirdCellCluster {
              primaryVertexContext.getClusters()[layerIndex + 2][nextTracklet.secondClusterIndex] };

          const float thirdCellClusterQuadraticRCoordinate { thirdCellCluster.rCoordinate * thirdCellCluster.rCoordinate };

          const std::array<float, 3> secondDeltaVector { { thirdCellCluster.xCoordinate - firstCellCluster.xCoordinate,
              thirdCellCluster.yCoordinate - firstCellCluster.yCoordinate, thirdCellClusterQuadraticRCoordinate
                  - firstCellClusterQuadraticRCoordinate } };

          std::array<float, 3> cellPlaneNormalVector { CAMathUtils::crossProduct(firstDeltaVector, secondDeltaVector) };

          const float vectorNorm { std::sqrt(
              cellPlaneNormalVector[0] * cellPlaneNormalVector[0] + cellPlaneNormalVector[1] * cellPlaneNormalVector[1]
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
          const std::array<float, 2> circleCenter { { -0.5f * normalizedPlaneVector[0] / normalizedPlaneVector[2], -0.5f
              * normalizedPlaneVector[1] / normalizedPlaneVector[2] } };
          const float distanceOfClosestApproach { std::abs(
              cellTrajectoryRadius - std::sqrt(circleCenter[0] * circleCenter[0] + circleCenter[1] * circleCenter[1])) };

          if (distanceOfClosestApproach
              > CAConstants::Thresholds::CellMaxDistanceOfClosestApproachThreshold()[layerIndex]) {

            continue;
          }

          const float cellTrajectoryCurvature { 1.0f / cellTrajectoryRadius };

          if (layerIndex > 0
              && primaryVertexContext.getCellsLookupTable()[layerIndex - 1][iTracklet]
                  == CAConstants::ITS::UnusedIndex) {

            primaryVertexContext.getCellsLookupTable()[layerIndex - 1][iTracklet] =
                primaryVertexContext.getCells()[layerIndex].size();
          }

          primaryVertexContext.getCells()[layerIndex].emplace_back(currentTracklet.firstClusterIndex,
              nextTracklet.firstClusterIndex, nextTracklet.secondClusterIndex, iTracklet, iNextLayerTracklet,
              normalizedPlaneVector, cellTrajectoryCurvature);
        }
      }
    }
  }
}

template<>
void CATrackerTraits<false>::postProcessCells(Context& primaryVertexContext)
{
  // Nothing to do
}

template<bool IsGPU>
CATracker<IsGPU>::CATracker(const CAEvent& event)
    : mEvent(event), mUsedClustersTable(event.getTotalClusters(), CAConstants::ITS::UnusedIndex)
{
  // Nothing to do
}

template<bool IsGPU>
std::vector<std::vector<CARoad>> CATracker<IsGPU>::clustersToTracks()
{
  const int verticesNum { mEvent.getPrimaryVerticesNum() };
  std::vector<std::vector<CARoad>> roads { };
  roads.reserve(verticesNum);

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    TrackerContext primaryVertexContext { mEvent, iVertex };

    computeTracklets(primaryVertexContext);
    computeCells(primaryVertexContext);
    findCellsNeighbours(primaryVertexContext);
    findTracks(primaryVertexContext);
    computeMontecarloLabels(primaryVertexContext);

    roads.emplace_back(primaryVertexContext.getRoads());
  }

  return roads;
}

template<bool IsGPU>
std::vector<std::vector<CARoad>> CATracker<IsGPU>::clustersToTracksVerbose()
{
  const int verticesNum { mEvent.getPrimaryVerticesNum() };
  std::vector<std::vector<CARoad>> roads { };
  roads.reserve(verticesNum);

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    clock_t t1 { }, t2 { };
    float diff { };

    t1 = clock();

    TrackerContext primaryVertexContext { mEvent, iVertex };

    evaluateTask(&CATracker<IsGPU>::computeTracklets, "Tracklets Finding", primaryVertexContext);
    evaluateTask(&CATracker<IsGPU>::computeCells, "Cells Finding", primaryVertexContext);
    evaluateTask(&CATracker<IsGPU>::findCellsNeighbours, "Neighbours Finding", primaryVertexContext);
    evaluateTask(&CATracker<IsGPU>::findTracks, "Tracks Finding", primaryVertexContext);
    evaluateTask(&CATracker<IsGPU>::computeMontecarloLabels, "Computing Montecarlo Labels", primaryVertexContext);

    t2 = clock();
    diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
    std::cout << std::setw(2) << " - Vertex " << iVertex + 1 << " completed in: " << diff << "ms" << std::endl;

    std::cout << "Found " << primaryVertexContext.getRoads().size() << " roads for vertex " << iVertex + 1 << std::endl;

    roads.emplace_back(primaryVertexContext.getRoads());
  }

  return roads;
}

template<bool IsGPU>
std::vector<std::vector<CARoad>> CATracker<IsGPU>::clustersToTracksMemoryBenchmark(
    std::ofstream & memoryBenchmarkOutputStream)
{
  const int verticesNum { mEvent.getPrimaryVerticesNum() };
  std::vector<std::vector<CARoad>> roads { };
  roads.reserve(verticesNum);

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    TrackerContext primaryVertexContext { mEvent, iVertex };

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.getClusters()[iLayer].size() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.getTracklets()[iLayer].capacity() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    computeTracklets(primaryVertexContext);

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.getTracklets()[iLayer].size() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.getCells()[iLayer].capacity() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    computeCells(primaryVertexContext);

    for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

      memoryBenchmarkOutputStream << primaryVertexContext.getCells()[iLayer].size() << "\t";
    }

    memoryBenchmarkOutputStream << std::endl;

    findCellsNeighbours(primaryVertexContext);
    findTracks(primaryVertexContext);
    computeMontecarloLabels(primaryVertexContext);

    roads.emplace_back(primaryVertexContext.getRoads());

    memoryBenchmarkOutputStream << primaryVertexContext.getRoads().size() << std::endl;
  }

  return roads;
}

template<bool IsGPU>
void CATracker<IsGPU>::computeTracklets(TrackerContext& primaryVertexContext)
{
  const std::array<float, 3>& primaryVertex { mEvent.getPrimaryVertex(primaryVertexContext.getPrimaryVertex()) };

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::TrackletsPerRoad; ++iLayer) {

    if (primaryVertexContext.getClusters()[iLayer].empty() || primaryVertexContext.getClusters()[iLayer + 1].empty()) {

      continue;
    }

    TrackerTraits::computeLayerTracklets(primaryVertexContext, iLayer, primaryVertex);
  }

  TrackerTraits::postProcessTracklets(primaryVertexContext);
}

template<bool IsGPU>
void CATracker<IsGPU>::computeCells(TrackerContext& primaryVertexContext)
{
  const std::array<float, 3>& primaryVertex { mEvent.getPrimaryVertex(primaryVertexContext.getPrimaryVertex()) };

  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad; ++iLayer) {

    TrackerTraits::computeLayerCells(primaryVertexContext, iLayer, primaryVertex);
  }

  TrackerTraits::postProcessCells(primaryVertexContext);
}

template<bool IsGPU>
void CATracker<IsGPU>::findCellsNeighbours(TrackerContext& primaryVertexContext)
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::CellsPerRoad - 1; ++iLayer) {

    if (primaryVertexContext.getCells()[iLayer + 1].empty()
        || primaryVertexContext.getCellsLookupTable()[iLayer].empty()) {

      continue;
    }

    int layerCellsNum { static_cast<int>(primaryVertexContext.getCells()[iLayer].size()) };

    for (int iCell { 0 }; iCell < layerCellsNum; ++iCell) {

      const CACell& currentCell { primaryVertexContext.getCells()[iLayer][iCell] };
      const int nextLayerTrackletIndex { currentCell.getSecondTrackletIndex() };
      const int nextLayerFirstCellIndex { primaryVertexContext.getCellsLookupTable()[iLayer][nextLayerTrackletIndex] };

      if (nextLayerFirstCellIndex == CAConstants::ITS::UnusedIndex) {

        continue;
      }

      const int nextLayerCellsNum { static_cast<int>(primaryVertexContext.getCells()[iLayer + 1].size()) };
      primaryVertexContext.getCellsNeighbours()[iLayer].resize(nextLayerCellsNum);

      for (int iNextLayerCell { nextLayerFirstCellIndex };
          iNextLayerCell < nextLayerCellsNum
              && primaryVertexContext.getCells()[iLayer + 1][iNextLayerCell].getFirstTrackletIndex()
                  == nextLayerTrackletIndex; ++iNextLayerCell) {

        CACell& nextCell { primaryVertexContext.getCells()[iLayer + 1][iNextLayerCell] };
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

          primaryVertexContext.getCellsNeighbours()[iLayer][iNextLayerCell].push_back(iCell);

          const int currentCellLevel { currentCell.getLevel() };

          if (currentCellLevel >= nextCell.getLevel()) {

            nextCell.setLevel(currentCellLevel + 1);
          }
        }
      }
    }
  }
}

template<bool IsGPU>
void CATracker<IsGPU>::findTracks(TrackerContext& primaryVertexContext)
{
  for (int iLevel { CAConstants::ITS::CellsPerRoad }; iLevel >= CAConstants::Thresholds::CellsMinLevel; --iLevel) {

    const int minimumLevel { iLevel - 1 };

    for (int iLayer { CAConstants::ITS::CellsPerRoad - 1 }; iLayer >= minimumLevel; --iLayer) {

      const int levelCellsNum { static_cast<int>(primaryVertexContext.getCells()[iLayer].size()) };

      for (int iCell { 0 }; iCell < levelCellsNum; ++iCell) {

        CACell& currentCell { primaryVertexContext.getCells()[iLayer][iCell] };

        if (currentCell.getLevel() != iLevel) {

          continue;
        }

        primaryVertexContext.getRoads().emplace_back(iLayer, iCell);

        const int cellNeighboursNum { static_cast<int>(primaryVertexContext.getCellsNeighbours()[iLayer - 1][iCell].size()) };

        for (int iNeighbourCell { 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

          if (iNeighbourCell > 0) {

            primaryVertexContext.getRoads().emplace_back(iLayer, iCell);
          }

          traverseCellsTree(primaryVertexContext, primaryVertexContext.getCellsNeighbours()[iLayer - 1][iCell][iNeighbourCell], iLayer - 1);
        }

        currentCell.setLevel(0);
      }
    }
  }
}

template<bool IsGPU>
void CATracker<IsGPU>::traverseCellsTree(TrackerContext& primaryVertexContext, const int currentCellId,
    const int currentLayerId)
{
  if (currentLayerId == 0) {

    return;
  }

  CACell& currentCell { primaryVertexContext.getCells()[currentLayerId][currentCellId] };

  primaryVertexContext.getRoads().back().addCell(currentLayerId, currentCellId);

  const int cellNeighboursNum { static_cast<int>(primaryVertexContext.getCellsNeighbours()[currentLayerId - 1][currentCellId].size()) };

  for (int iNeighbourCell { 0 }; iNeighbourCell < cellNeighboursNum; ++iNeighbourCell) {

    if (iNeighbourCell > 0) {

      primaryVertexContext.getRoads().push_back(primaryVertexContext.getRoads().back());

    }

    traverseCellsTree(primaryVertexContext, primaryVertexContext.getCellsNeighbours()[currentLayerId - 1][currentCellId][iNeighbourCell], currentLayerId - 1);
  }

  currentCell.setLevel(0);
}

template<bool IsGPU>
void CATracker<IsGPU>::computeMontecarloLabels(TrackerContext& primaryVertexContext)
{
/// Moore’s Voting Algorithm

  int roadsNum { static_cast<int>(primaryVertexContext.getRoads().size()) };

  for (int iRoad { 0 }; iRoad < roadsNum; ++iRoad) {

    CARoad& currentRoad { primaryVertexContext.getRoads()[iRoad] };
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

      const CACell& currentCell { primaryVertexContext.getCells()[iCell][currentCellIndex] };

      if (isFirstRoadCell) {

        maxOccurrencesValue =
            primaryVertexContext.getClusters()[iCell][currentCell.getFirstClusterIndex()].monteCarloId;
        count = 1;

        const int secondMonteCarlo {
            primaryVertexContext.getClusters()[iCell + 1][currentCell.getSecondClusterIndex()].monteCarloId };

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
          primaryVertexContext.getClusters()[iCell + 2][currentCell.getThirdClusterIndex()].monteCarloId };

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

template<bool IsGPU>
void CATracker<IsGPU>::evaluateTask(void (CATracker<IsGPU>::*task)(TrackerContext&), const char *taskName,
    TrackerContext& primaryVertexContext)
{
  clock_t t1 { }, t2 { };
  float diff { };

  t1 = clock();

  (this->*task)(primaryVertexContext);

  t2 = clock();
  diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
  std::cout << std::setw(2) << " - " << taskName << " completed in: " << diff << "ms" << std::endl;
}

template class CATracker<TRACKINGITSU_GPU_MODE> ;
