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
    CAIndexTable(const CALayer&, const int);

    const std::vector<int> getBin(const int) const;

    const std::vector<int> selectBins(const float, const float, const float, const float);

  private:
    int mLayerIndex;
    std::array<std::vector<int>, CAConstants::IndexTable::ZBins * CAConstants::IndexTable::PhiBins + 1> mTableBins;
};

inline const std::vector<int> CAIndexTable::getBin(const int binIndex) const
{

  return mTableBins[binIndex];
}

#endif /* TRACKINGITSU_INCLUDE_CALOOKUPTABLE_H_ */
