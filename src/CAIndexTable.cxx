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

CAIndexTable::CAIndexTable()
    : mLayerMinZCoordinate { }, mLayerMaxZCoordinate { }, mInverseZBinSize { }, mInversePhiBinSize { }, mTableBins { }
{
  //Dummy constructor
}

CAIndexTable::CAIndexTable(const CALayer& layer)
    : mLayerMinZCoordinate { layer.getMinZCoordinate() }, mLayerMaxZCoordinate { layer.getMaxZCoordinate() }, mInverseZBinSize {
        CAConstants::LookupTable::ZBins / (layer.getMaxZCoordinate() - layer.getMinZCoordinate()) }, mInversePhiBinSize {
        CAConstants::LookupTable::PhiBins / CAConstants::Math::TwoPi }
{
  int layerClustersNum = layer.getClustersSize();

  for (int iCluster = 0; iCluster < layerClustersNum; ++iCluster) {

    const CACluster& currentCluster = layer.getCluster(iCluster);

    const int currentBinIndex = getBinIndex(getZBinIndex(currentCluster.zCoordinate),
        getPhiBinIndex(currentCluster.phiCoordinate));
    mTableBins[currentBinIndex].push_back(iCluster);
  }
}

std::vector<int> CAIndexTable::selectClusters(const float zRangeMin, const float zRangeMax, const float phiRangeMin,
    const float phiRangeMax)
{
  std::vector<int> filteredClusters;

  if (zRangeMax < mLayerMinZCoordinate || zRangeMin > mLayerMaxZCoordinate || zRangeMin > zRangeMax) {

    return filteredClusters;
  }

  const int minZBinIndex = std::max(0, getZBinIndex(zRangeMin));
  const int maxZBinIndex = std::min(CAConstants::LookupTable::ZBins, getZBinIndex(zRangeMax));
  const int zBinsNum = maxZBinIndex - minZBinIndex + 1;
  const int minPhiBinIndex = getPhiBinIndex(phiRangeMin);
  const int maxPhiBinIndex = getPhiBinIndex(phiRangeMax);

  for (int iPhiBin = minPhiBinIndex; iPhiBin <= maxPhiBinIndex;
      iPhiBin = iPhiBin + 1 % CAConstants::LookupTable::PhiBins) {

    for (int iBinIndex = getBinIndex(minZBinIndex, iPhiBin); iBinIndex <= zBinsNum; ++iBinIndex) {

      if (!mTableBins[iBinIndex].empty()) {

        filteredClusters.insert(std::end(filteredClusters), std::begin(mTableBins[iBinIndex]),
            std::end(mTableBins[iBinIndex]));
      }
    }
  }

  return filteredClusters;
}
