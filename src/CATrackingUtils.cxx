/// \file CATrackingUtils.cxx
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

#include "CATrackingUtils.h"

#include <cmath>

#include "CAConstants.h"
#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"

namespace TRACKINGITSU_TARGET_NAMESPACE {
GPU_DEVICE bool CATrackingUtils::isValidTracklet(const CACluster &firstLayerCluster, const CACluster &secondLayerCluster,
    const float tanLambda, const float directionZIntersection)
{
  const float deltaZ { MATH_ABS(
      tanLambda * (secondLayerCluster.rCoordinate - firstLayerCluster.rCoordinate) + firstLayerCluster.zCoordinate
          - secondLayerCluster.zCoordinate) };
  const float deltaPhi { MATH_ABS(firstLayerCluster.phiCoordinate - secondLayerCluster.phiCoordinate) };

  return deltaZ < CAConstants::Thresholds::TrackletMaxDeltaZThreshold()[firstLayerCluster.layerIndex]
      && (deltaPhi < CAConstants::Thresholds::PhiCoordinateCut
          || MATH_ABS(deltaPhi - CAConstants::Math::TwoPi) < CAConstants::Thresholds::PhiCoordinateCut);
}

GPU_DEVICE const GPU_ARRAY<int, 4> CATrackingUtils::getBinsRect(const CACluster& currentCluster, const int layerIndex,
    const float directionZIntersection)
{
  const float zRangeMin = directionZIntersection - 2 * CAConstants::Thresholds::ZCoordinateCut;
  const float phiRangeMin = currentCluster.phiCoordinate - CAConstants::Thresholds::PhiCoordinateCut;
  const float zRangeMax = directionZIntersection + 2 * CAConstants::Thresholds::ZCoordinateCut;
  const float phiRangeMax = currentCluster.phiCoordinate + CAConstants::Thresholds::PhiCoordinateCut;

  if (zRangeMax < -CAConstants::ITS::LayersZCoordinate()[layerIndex + 1]
      || zRangeMin > CAConstants::ITS::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return EmptyBinsRect;
  }

  return GPU_ARRAY<int, 4> { { MATH_MAX(0, CAIndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
      CAIndexTableUtils::getPhiBinIndex(CAMathUtils::getNormalizedPhiCoordinate(phiRangeMin)), MATH_MIN(
          CAConstants::IndexTable::ZBins - 1, CAIndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
      CAIndexTableUtils::getPhiBinIndex(CAMathUtils::getNormalizedPhiCoordinate(phiRangeMax)) } };
}
}
