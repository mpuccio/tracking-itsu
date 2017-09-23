/// \file Cell.h
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

#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class Cell
  final
  {
    public:
      GPU_DEVICE Cell(const int, const int, const int, const int, const int, const float3&, const float);

      int getFirstClusterIndex() const;
      int getSecondClusterIndex() const;
      int getThirdClusterIndex() const;
      GPU_HOST_DEVICE int getFirstTrackletIndex() const;
      int getSecondTrackletIndex() const;
      int getLevel() const;
      float getCurvature() const;
      const float3& getNormalVectorCoordinates() const;
      void setLevel(const int level);

    private:
      const int mFirstClusterIndex;
      const int mSecondClusterIndex;
      const int mThirdClusterIndex;
      const int mFirstTrackletIndex;
      const int mSecondTrackletIndex;
      const float3 mNormalVectorCoordinates;
      const float mCurvature;
      int mLevel;
  };

  inline int Cell::getFirstClusterIndex() const
  {
    return mFirstClusterIndex;
  }

  inline int Cell::getSecondClusterIndex() const
  {
    return mSecondClusterIndex;
  }

  inline int Cell::getThirdClusterIndex() const
  {
    return mThirdClusterIndex;
  }

  GPU_HOST_DEVICE inline int Cell::getFirstTrackletIndex() const
  {
    return mFirstTrackletIndex;
  }

  inline int Cell::getSecondTrackletIndex() const
  {
    return mSecondTrackletIndex;
  }

  inline int Cell::getLevel() const
  {
    return mLevel;
  }

  inline float Cell::getCurvature() const
  {
    return mCurvature;
  }

  inline const float3& Cell::getNormalVectorCoordinates() const
  {
    return mNormalVectorCoordinates;
  }

  inline void Cell::setLevel(const int level)
  {
    mLevel = level;
  }

}
}
}
#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
