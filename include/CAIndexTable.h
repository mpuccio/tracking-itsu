/// \file CAIndexTable.h
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

#ifndef TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_
#define TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_

#include <array>

#include "CAConstants.h"
#include "CALayer.h"
#include "CAMathUtils.h"

class CAIndexTable final
{
  public:
    CAIndexTable();
    explicit CAIndexTable(const CALayer&);

    int getZBinIndex(const float) const;
    int getPhiBinIndex(const float) const;
    int getBinIndex(const int, const int) const;
    const std::vector<std::reference_wrapper<std::vector<int>>> selectClusters(const float, const float, const float, const float);

  private:
    float mLayerMinZCoordinate;
    float mLayerMaxZCoordinate;
    float mInverseZBinSize;
    float mInversePhiBinSize;
    std::array<std::vector<int>, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1> mTableBins;
};

inline int CAIndexTable::getZBinIndex(const float zCoordinate) const
{

  return (zCoordinate - mLayerMinZCoordinate) * mInverseZBinSize;
}

inline int CAIndexTable::getPhiBinIndex(const float currentPhi) const
{

  return (currentPhi * mInversePhiBinSize);
}

inline int CAIndexTable::getBinIndex(const int zIndex, const int phiIndex) const
{

  return phiIndex * CAConstants::IndexTable::PhiBins + zIndex;
}

#endif /* TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_ */
