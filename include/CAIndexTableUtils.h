/// \file CAIndexTableUtils.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

/***************************************************************************
 *  Copyright (C) 2017  Iacopo Colonnelli
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/
#ifndef TRACKINGITSU_INCLUDE_CAINDEXTABLEUTILS_H_
#define TRACKINGITSU_INCLUDE_CAINDEXTABLEUTILS_H_

#include "CAConstants.h"

namespace CAIndexTableUtils {
constexpr float getInverseZBinSize(const int);
constexpr int getZBinIndex(const float, const float);
constexpr int getPhiBinIndex(const float);
constexpr int getBinIndex(const int, const int);
}
;

constexpr float getInverseZCoordinate(const int layerIndex)
{

  return 0.5f * CAConstants::IndexTable::ZBins / CAConstants::ITS::LayersZCoordinate[layerIndex];
}

constexpr int CAIndexTableUtils::getZBinIndex(const float layerIndex, const float zCoordinate)
{

  return (zCoordinate + CAConstants::ITS::LayersZCoordinate[layerIndex])
      * CAConstants::IndexTable::InverseZBinSize[layerIndex];
}

constexpr int CAIndexTableUtils::getPhiBinIndex(const float currentPhi)
{

  return (currentPhi * CAConstants::IndexTable::InversePhiBinSize);
}

constexpr int CAIndexTableUtils::getBinIndex(const int zIndex, const int phiIndex)
{
  return std::min(phiIndex * CAConstants::IndexTable::PhiBins + zIndex,
      CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins);
}

#endif /* TRACKINGITSU_INCLUDE_CAINDEXTABLEUTILS_H_ */
