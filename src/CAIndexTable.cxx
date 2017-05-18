/// \file CAIndexTable.cxx
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

#include "CAIndexTable.h"

#include <algorithm>

#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"

CAIndexTable::CAIndexTable()
    : mLayerIndex { CAConstants::ITS::UnusedIndex }
{
  // Nothing to do
}

CAIndexTable::CAIndexTable(const int layerIndex, const std::vector<CACluster>& clusters)
    : mLayerIndex { layerIndex }
{
  const int layerClustersNum { static_cast<int>(clusters.size()) };
  int previousBinIndex { 0 };
  mTableBins[0] = 0;

  for (int iCluster { 1 }; iCluster < layerClustersNum; ++iCluster) {

    const int currentBinIndex { clusters[iCluster].indexTableBinIndex };

    if (currentBinIndex > previousBinIndex) {

      for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

        mTableBins[iBin] = iCluster;
      }

      previousBinIndex = currentBinIndex;
    }
  }

  for (int iBin { previousBinIndex + 1 }; iBin <= CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins;
      iBin++) {

    mTableBins[iBin] = layerClustersNum;
  }

}

const std::array<int, 4> CAIndexTable::getSelectedBinsRect(const float zRangeMin, const float phiRangeMin,
    const float zRangeMax, const float phiRangeMax) const
{
  return std::array<int, 4> { { std::max(0, CAIndexTableUtils::getZBinIndex(mLayerIndex, zRangeMin)),
      CAIndexTableUtils::getPhiBinIndex(CAMathUtils::getNormalizedPhiCoordinate(phiRangeMin)), std::min(
          CAConstants::IndexTable::ZBins - 1, CAIndexTableUtils::getZBinIndex(mLayerIndex, zRangeMax)),
      CAIndexTableUtils::getPhiBinIndex(CAMathUtils::getNormalizedPhiCoordinate(phiRangeMax)) } };
}

const std::vector<std::pair<int, int>> CAIndexTable::selectClusters(const std::array<int, 4>& selectedBinsRect) const
{
  std::vector<std::pair<int, int>> filteredBins { };

  const int zBinsNum { selectedBinsRect[2] - selectedBinsRect[0] + 1 };
  int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

  if (phiBinsNum < 0) {

    phiBinsNum += CAConstants::IndexTable::PhiBins;
  }

  filteredBins.reserve(phiBinsNum);

  for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
      iPhiBin = ++iPhiBin == CAConstants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

    const int firstBinIndex { CAIndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };
    const int maxBinIndex { firstBinIndex + zBinsNum };

    filteredBins.emplace_back(mTableBins[firstBinIndex], mTableBins[maxBinIndex] - mTableBins[firstBinIndex] + 1);
  }

  return filteredBins;
}
