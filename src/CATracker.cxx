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

GPU_HOST_DEVICE bool CABaseTrackerTraits::isValidTracklet(const CACluster &firstLayerCluster,
    const CACluster &secondLayerCluster, const float tanLambda, const float directionZIntersection)
{
  const float deltaZ { MATH_ABS(
      tanLambda * (secondLayerCluster.rCoordinate - firstLayerCluster.rCoordinate) + firstLayerCluster.zCoordinate
          - secondLayerCluster.zCoordinate) };
  const float deltaPhi { MATH_ABS(firstLayerCluster.phiCoordinate - secondLayerCluster.phiCoordinate) };

  return deltaZ < CAConstants::Thresholds::TrackletMaxDeltaZThreshold()[firstLayerCluster.layerIndex]
      && (deltaPhi < CAConstants::Thresholds::PhiCoordinateCut
          || MATH_ABS(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::PhiCoordinateCut);
}

template<>
void CATrackerTraits<false>::getTrackletsFromCluster(Context& primaryVertexContext, const int iLayer,
    const int iCluster, const float tanLambda, const float directionZIntersection,
    const std::array<int, 4> selectedBinsRect, const std::vector<std::pair<int, int>> nextLayerClustersSubset)
{
  const CACluster& currentCluster = primaryVertexContext.clusters[iLayer][iCluster];
  const int rowsNum { static_cast<int>(nextLayerClustersSubset.size()) };

  for (int iRow { 0 }; iRow < rowsNum; ++iRow) {

    const int firstRowClusterIndex { nextLayerClustersSubset[iRow].first };
    const int lastRowClusterIndex { firstRowClusterIndex + nextLayerClustersSubset[iRow].second - 1 };

    for (int iNextLayerCluster { firstRowClusterIndex }; iNextLayerCluster <= lastRowClusterIndex;
        ++iNextLayerCluster) {

      const CACluster& nextCluster { primaryVertexContext.clusters[iLayer + 1][iNextLayerCluster] };

      if (isValidTracklet(currentCluster, nextCluster, tanLambda, directionZIntersection)) {

        if (iLayer > 0
            && primaryVertexContext.trackletsLookupTable[iLayer - 1][iCluster] == CAConstants::ITS::UnusedIndex) {

          primaryVertexContext.trackletsLookupTable[iLayer - 1][iCluster] =
              primaryVertexContext.tracklets[iLayer].size();
        }

        primaryVertexContext.tracklets[iLayer].emplace_back(iCluster, iNextLayerCluster, currentCluster, nextCluster);
      }
    }
  }
}
}
