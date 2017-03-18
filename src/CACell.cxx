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

CACell::CACell(const std::array<int, 3>& trackletCoordinates, const std::array<int, 2>& doubletsIds,
    const std::array<float, 3>& normalVectorCoordinates, const float curvature)
    : mTrackletCoordinates(trackletCoordinates), mTrackletIds(doubletsIds), mNormalVectorCoordinates(
        normalVectorCoordinates), mCurvature { curvature }, mLevel { 1 }, mNeighbours { }
{
  // Nothing to do
}

inline const int CACell::getXCoordinate() const
{
  return mTrackletCoordinates[0];
}

inline const int CACell::getYCoordinate() const
{
  return mTrackletCoordinates[1];
}

inline const int CACell::getZCoordinate() const
{
  return mTrackletCoordinates[2];
}

inline const int CACell::getLevel() const
{
  return mLevel;
}

inline const float CACell::getCurvature() const
{
  return mCurvature;
}

inline const int CACell::getNumberOfNeighbours() const
{
  return mNeighbours.size();
}

inline const std::array<float, 3>& CACell::getNormalVectorCoordinates() const
{
  return mNormalVectorCoordinates;
}

void CACell::setLevel(const int level)
{
  this->mLevel = level;
}

bool CACell::combineCells(const CACell& otherCell, int otherCellId)
{
  if (this->getYCoordinate() == otherCell.getZCoordinate() && this->getXCoordinate() == otherCell.getYCoordinate()) {

    mNeighbours.push_back(otherCellId);

    int otherCellLevel = otherCell.getLevel();

    if (otherCellLevel >= getLevel()) {

      setLevel(otherCellLevel + 1);
    }

    return true;

  } else {

    return false;
  }
}
