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

#include <array>
#include <utility>
#include <vector>

#include "CAConstants.h"
#include "CADefinitions.h"

namespace CAIndexTableUtils {
float getInverseZBinSize(const int);
GPU_HOST_DEVICE int getZBinIndex(const int, const float);
GPU_HOST_DEVICE int getPhiBinIndex(const float);
GPU_HOST_DEVICE int getBinIndex(const int, const int);
GPU_HOST_DEVICE int countRowSelectedBins(
    const GPUArray<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>&, const int, const int,
    const int);
const std::vector<std::pair<int, int>> selectClusters(
    const std::array<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1>&,
    const std::array<int, 4>& selectedBinsRect);
}

inline float getInverseZCoordinate(const int layerIndex)
{
  return 0.5f * CAConstants::IndexTable::ZBins / CAConstants::ITS::LayersZCoordinate()[layerIndex];
}

GPU_HOST_DEVICE inline int CAIndexTableUtils::getZBinIndex(const int layerIndex, const float zCoordinate)
{
  return (zCoordinate + CAConstants::ITS::LayersZCoordinate()[layerIndex])
      * CAConstants::IndexTable::InverseZBinSize()[layerIndex];
}

GPU_HOST_DEVICE inline int CAIndexTableUtils::getPhiBinIndex(const float currentPhi)
{
  return (currentPhi * CAConstants::IndexTable::InversePhiBinSize);
}

GPU_HOST_DEVICE inline int CAIndexTableUtils::getBinIndex(const int zIndex, const int phiIndex)
{
  return MATH_MIN(phiIndex * CAConstants::IndexTable::PhiBins + zIndex,
      CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins);
}

GPU_HOST_DEVICE inline int CAIndexTableUtils::countRowSelectedBins(
    const GPUArray<int, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1> &indexTable,
    const int phiBinIndex, const int minZBinIndex, const int maxZBinIndex)
{
  const int firstBinIndex { getBinIndex(minZBinIndex, phiBinIndex) };
  const int maxBinIndex { firstBinIndex + maxZBinIndex - minZBinIndex + 1 };

  return indexTable[maxBinIndex] - indexTable[firstBinIndex] + 1;
}

#endif /* TRACKINGITSU_INCLUDE_CAINDEXTABLEUTILS_H_ */
