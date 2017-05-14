/// \file CACell.cxx
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

#include "CACell.h"

CACell::CACell(const int firstClusterIndex, const int secondClusterIndex, const int thirdClusterIndex,
    const int firstTrackletIndex, const int secondTrackletIndex, const std::array<float, 3>& normalVectorCoordinates,
    const float curvature)
    : mFirstClusterIndex { firstClusterIndex }, mSecondClusterIndex { secondClusterIndex }, mThirdClusterIndex {
        thirdClusterIndex }, mFirstTrackletIndex(firstTrackletIndex), mSecondTrackletIndex(secondTrackletIndex), mNormalVectorCoordinates(
        normalVectorCoordinates), mCurvature { curvature }, mLevel { 1 }
{
  // Nothing to do
}

bool CACell::combineCells(const CACell& otherCell, int otherCellId)
{
  if (mSecondClusterIndex == otherCell.getThirdClusterIndex()
      && mFirstClusterIndex == otherCell.getSecondClusterIndex()) {

    mNeighbours.push_back(otherCellId);

    const int otherCellLevel { otherCell.getLevel() };

    if (otherCellLevel >= mLevel) {

      setLevel(otherCellLevel + 1);
    }

    return true;

  } else {

    return false;
  }
}
