/// \file TrackingUtils.cxx
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

#include "ITSReconstruction/CA/TrackingUtils.h"

#include <cmath>

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"
#include "ITSReconstruction/CA/MathUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

GPU_DEVICE const int4 TrackingUtils::getBinsRect(const Cluster& currentCluster, const int layerIndex,
    const float directionZIntersection)
{
  const float zRangeMin = directionZIntersection - 2 * Constants::Thresholds::ZCoordinateCut;
  const float phiRangeMin = currentCluster.phiCoordinate - Constants::Thresholds::PhiCoordinateCut;
  const float zRangeMax = directionZIntersection + 2 * Constants::Thresholds::ZCoordinateCut;
  const float phiRangeMax = currentCluster.phiCoordinate + Constants::Thresholds::PhiCoordinateCut;

  if (zRangeMax < -Constants::ITS::LayersZCoordinate()[layerIndex + 1]
      || zRangeMin > Constants::ITS::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4 { MATH_MAX(0, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
      IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMin)), MATH_MIN(
          Constants::IndexTable::ZBins - 1, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
      IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMax)) };
}

}
}
}
