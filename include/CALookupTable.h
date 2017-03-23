/// \file CALookupTable.h
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

class CALookupTable final
{
  public:
    CALookupTable();
    explicit CALookupTable(const CALayer&);

    int getZBinIndex(const float) const;
    int getPhiBinIndex(const float) const;
    int getBinIndex(const int, const int) const;
    std::vector<int> selectClusters(const float, const float, const float, const float);

  private:
    float mLayerMinZCoordinate;
    float mLayerMaxZCoordinate;
    float mInverseZBinSize;
    float mInversePhiBinSize;
    std::array<std::vector<int>, CAConstants::LookupTable::ZBins * CAConstants::LookupTable::PhiBins> mTableBins;
};

inline int CALookupTable::getZBinIndex(const float zCoordinate) const
{

  return (zCoordinate - mLayerMinZCoordinate) * mInverseZBinSize;
}

inline int CALookupTable::getPhiBinIndex(const float currentPhi) const
{

  return (CAMathUtils::getNormalizedPhiCoordinate(currentPhi) * mInversePhiBinSize);
}

inline int CALookupTable::getBinIndex(const int phiIndex, const int zIndex) const
{

  return phiIndex * CAConstants::LookupTable::PhiBins + zIndex;
}

#endif /* TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_ */
