/// \file CACell.h
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

#ifndef TRACKINGITSU_INCLUDE_CACELL_H_
#define TRACKINGITSU_INCLUDE_CACELL_H_

#include <array>
#include <vector>

class CACell final
{
  public:
    CACell(const std::array<int, 3>&, const std::array<int, 2>&, const std::array<float, 3>&, const float);

    int getXCoordinate() const;
    int getYCoordinate() const;
    int getZCoordinate() const;
    int getLevel() const;
    float getCurvature() const;
    int getNumberOfNeighbours() const;
    const std::array<float, 3>& getNormalVectorCoordinates() const;

    void setLevel(const int level);

    bool combineCells(const CACell&, int);

  private:
    const std::array<int, 3> mTrackletCoordinates;
    const std::array<int, 2> mTrackletIds;
    const std::array<float, 3> mNormalVectorCoordinates;
    const float mCurvature;
    int mLevel;
    std::vector<int> mNeighbours;
};

inline int CACell::getXCoordinate() const
{
  return mTrackletCoordinates[0];
}

inline int CACell::getYCoordinate() const
{
  return mTrackletCoordinates[1];
}

inline int CACell::getZCoordinate() const
{
  return mTrackletCoordinates[2];
}

inline int CACell::getLevel() const
{
  return mLevel;
}

inline float CACell::getCurvature() const
{
  return mCurvature;
}

inline int CACell::getNumberOfNeighbours() const
{
  return mNeighbours.size();
}

inline const std::array<float, 3>& CACell::getNormalVectorCoordinates() const
{
  return mNormalVectorCoordinates;
}

#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
