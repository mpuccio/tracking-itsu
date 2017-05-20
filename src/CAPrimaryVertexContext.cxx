/// \file CAPrimaryVertexContext.cxx
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

#include "CAPrimaryVertexContext.h"

#include <algorithm>
#include <cmath>

#include "CAConstants.h"
#include "CAEvent.h"
#include "CALayer.h"

namespace {
CAPrimaryVertexContextTraits::CAPrimaryVertexContextTraits(const CAEvent& event, const int primaryVertexIndex)
    : primaryVertexIndex { primaryVertexIndex }
{
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

    if (iLayer > 0) {

      indexTables[iLayer - 1] = CAIndexTable(iLayer, clusters[iLayer]);
    }

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      tracklets[iLayer].reserve(
          std::ceil(
              (CAConstants::Memory::TrackletsMemoryCoefficients[iLayer] * clustersNum)
                  * event.getLayer(iLayer + 1).getClustersSize()));
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      trackletsLookupTable[iLayer].resize(event.getLayer(iLayer + 1).getClustersSize(), CAConstants::ITS::UnusedIndex);

      cells[iLayer].reserve(
          std::ceil(
              ((CAConstants::Memory::CellsMemoryCoefficients[iLayer] * clustersNum)
                  * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize()));
    }
  }
}
}

CAGPUPrimaryVertexContext::CAGPUPrimaryVertexContext(const CAPrimaryVertexContextTraits& primaryVertexContextTraits)
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    clusters[iLayer] = CAGPUVector<CACluster> { &primaryVertexContextTraits.clusters[iLayer][0],
        static_cast<int>(primaryVertexContextTraits.clusters[iLayer].size()) };

    if (iLayer > 0) {

      indexTables[iLayer - 1] = CAGPUVector<int> { primaryVertexContextTraits.indexTables[iLayer - 1].getTable().data(),
          CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1 };
    }

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      tracklets[iLayer] = CAGPUVector<CATracklet> {
          static_cast<int>(primaryVertexContextTraits.tracklets[iLayer].capacity()) };
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      trackletsLookupTable[iLayer] = CAGPUVector<int> { &primaryVertexContextTraits.trackletsLookupTable[iLayer][0],
          static_cast<int>(primaryVertexContextTraits.trackletsLookupTable[iLayer].size()) };
    }
  }
}

CAGPUPrimaryVertexContext::~CAGPUPrimaryVertexContext()
{
  for (int iLayer { 0 }; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    clusters[iLayer].destroy();

    if (iLayer < CAConstants::ITS::TrackletsPerRoad) {

      indexTables[iLayer].destroy();
      tracklets[iLayer].destroy();
    }

    if (iLayer < CAConstants::ITS::CellsPerRoad) {

      trackletsLookupTable[iLayer].destroy();
    }
  }
}

CAPrimaryVertexContext<true>::CAPrimaryVertexContext(const CAEvent &event, const int primaryVertexContext)
    : CAPrimaryVertexContextTraits { event, primaryVertexContext }, gpuContext { *this }
{
  try {

    CAGPUUtils::Host::gpuMalloc(reinterpret_cast<void**>(&gpuContextDevicePointer), sizeof(CAGPUPrimaryVertexContext));
    CAGPUUtils::Host::gpuMemcpyHostToDevice(gpuContextDevicePointer, &gpuContext, sizeof(CAGPUPrimaryVertexContext));

  } catch(...) {

    CAGPUUtils::Host::gpuFree(gpuContextDevicePointer);
  }
}

CAPrimaryVertexContext<true>::~CAPrimaryVertexContext()
{
  CAGPUUtils::Host::gpuFree(gpuContextDevicePointer);
}
