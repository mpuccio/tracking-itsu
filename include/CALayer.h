/// \file CALayer.h
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
#ifndef INCLUDE_CALAYER_H_
#define INCLUDE_CALAYER_H_

#include <vector>

#include "CACluster.h"

class CALayer final
{
  public:
    CALayer();

    const std::vector<CACluster>& getClusters() const;
    const CACluster& getCluster(int) const;
    const int getClustersSize() const;
    const float getMinZCoordinate() const;
    const float getMaxZCoordinate() const;

    void addCluster(const int, const float, const float, const float, const int);

  private:
    std::vector<CACluster> mClusters;
    float mMinZCoordinate;
    float mMaxZCoordinate;
};

inline const std::vector<CACluster>& CALayer::getClusters() const
{
  return mClusters;
}

inline const CACluster& CALayer::getCluster(int clusterIndex) const
{
  return mClusters[clusterIndex];
}

inline const int CALayer::getClustersSize() const
{
  return mClusters.size();
}

inline const float CALayer::getMinZCoordinate() const
{
  return mMinZCoordinate;
}

inline const float CALayer::getMaxZCoordinate() const
{
  return mMaxZCoordinate;
}

#endif /* INCLUDE_CALAYER_H_ */
