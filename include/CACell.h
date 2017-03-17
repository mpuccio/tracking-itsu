/// \file CACell.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

/***************************************************************************
 *  Copyright (C) 2017  Iacopo Colonnelli
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#ifndef TRACKINGITSU_INCLUDE_CACELL_H_
#define TRACKINGITSU_INCLUDE_CACELL_H_

#include <array>
#include <vector>

class CACell final
{
  public:
    CACell(const std::array<int, 3>&, const std::array<int, 2>&, const std::array<float, 3>&, const float);

    const int getXCoordinate() const;
    const int getYCoordinate() const;
    const int getZCoordinate() const;
    const int getLevel() const;
    const float getCurvature() const;
    const int getNumberOfNeighbours() const;
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

#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
