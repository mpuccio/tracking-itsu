/// \file Road.h
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

#ifndef TRACKINGITSU_INCLUDE_ROAD_H_
#define TRACKINGITSU_INCLUDE_ROAD_H_

#include <array>

#include "ITSReconstruction/CA/Constants.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class Road
  final
  {
    public:
      Road();
      Road(int, int);

      int getRoadSize() const;
      int getLabel() const;
      void setLabel(const int);
      bool isFakeRoad() const;
      void setFakeRoad(const bool);
      int &operator[](const int&);

      void resetRoad();
      void addCell(int, int);

    private:
      std::array<int, Constants::ITS::CellsPerRoad> mCellIds;
      int mRoadSize;
      int mLabel;
      bool mIsFakeRoad;
  };

  inline int Road::getRoadSize() const
  {

    return mRoadSize;
  }

  inline int Road::getLabel() const
  {

    return mLabel;
  }

  inline void Road::setLabel(const int label)
  {
    mLabel = label;
  }

  inline int& Road::operator [](const int& i)
  {
    return mCellIds[i];
  }

  inline bool Road::isFakeRoad() const
  {
    return mIsFakeRoad;
  }

  inline void Road::setFakeRoad(const bool isFakeRoad)
  {
    mIsFakeRoad = isFakeRoad;
  }

}
}
}

#endif /* TRACKINGITSU_INCLUDE_ROAD_H_ */