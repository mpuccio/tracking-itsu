/// \file CALookupTable.cpp
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
#include "CALookupTable.h"

CALookupTable::CALookupTable(const CALayer& layer) : mLayer{layer},
  mInverseZBinSize{ITSConstants::LookupTableZBins / (layer.getMaxZCoordinate() - layer.getMinZCoordinate())},
  mInversePhiBinSize{ITSConstants::LookupTablePhiBins / MathConstants::TwoPi}
{
  int layerClustersNum = layer.getClustersSize();

  for(int iCluster = 0; iCluster < layerClustersNum; ++iCluster) {

    CACluster& currentCluster = layer.getCluster(iCluster);

    const int currentBinIndex = getBinIndex(getZBinIndex(currentCluster.zCoordinate), getPhiBinIndex(currentCluster.phiCoordinate));
    mTableBins[currentBinIndex].push_back(iCluster);
  }
}

std::vector<int> CALookupTable::selectClusters(const float phiRangeMin, const float phiRangeMax,
    const float zRangeMin, const float zRangeMax)
{
  std::vector<int> filteredClusters;

  if(zRangeMax < mLayer.getMinZCoordinate() || zRangeMin > mLayer.getMaxZCoordinate() || zRangeMin > zRangeMax) {

    return filteredClusters;
  }

  const int minZBinIndex = std::min(0, getZBinIndex(zRangeMin));
  const int maxZBinIndex = std::max(ITSConstants::LookupTableZBins - 1, getZBinIndex(zRangeMax));
  const int zBinsNum = maxZBinIndex - minZBinIndex + 1;
  const int minPhiBinIndex = getPhiBinIndex(phiRangeMin);
  const int maxPhiBinIndex = getPhiBinIndex(phiRangeMax);

  for(int iPhiBin = minPhiBinIndex; iPhiBin <= maxPhiBinIndex; ++iPhiBin % ITSConstants::LookupTablePhiBins) {

    for(int iBinIndex = getBinIndex(minZBinIndex, iPhiBin); iBinIndex <= zBinsNum; ++iBinIndex) {

      if(!mTableBins[iBinIndex].empty()) {

         filteredClusters.insert(std::end(filteredClusters), std::begin(mTableBins[iBinIndex]), std::end(mTableBins[iBinIndex]));
      }
  }

  return filteredClusters;
}
