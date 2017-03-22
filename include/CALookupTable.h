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

class CALookupTable final
{
  public:
    explicit CALookupTable(const CALayer&);

    int getZBinIndex(const float);
    int getPhiBinIndex(const float);
    int getBinIndex(const int, const int);
    std::vector<int> selectClusters(const float, const float, const float, const float);

  private:
    CALayer& mLayer;
    float mInverseZBinSize;
    float mInversePhiBinSize;
    std::array<std::vector<int>, ITSConstants::LookupTablePhiBins * ITSConstants::LookupTableZBins> mTableBins;
};

inline int CALookupTable::getZBinIndex(const float zCoordinate)
{

  return (zCoordinate - mLayer.getMinZCoordinate()) * mInverseZBinSize;
}

inline int CALookupTable::getPhiBinIndex(const float currentPhi)
{

  return (MathUtils::getNormalizedPhiCoordinate(currentPhi) * mInversePhiBinSize);
}

inline int CALookupTable::getBinIndex(const int phiIndex, const int zIndex) {

  return phiIndex * ITSConstants::LookupTablePhiBins + zIndex;
}

#endif /* TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_ */
