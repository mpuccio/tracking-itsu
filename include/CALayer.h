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
#ifndef TRACKINGITSU_INCLUDE_CALAYER_H_
#define TRACKINGITSU_INCLUDE_CALAYER_H_

#include <vector>

#include "CACluster.h"

class CALayer final
{
  public:
    CALayer();
    CALayer(const int);

    int getLayerIndex() const;
    const std::vector<CACluster>& getClusters() const;
    const CACluster& getCluster(int) const;
    int getClustersSize() const;

    void addCluster(const int, const float, const float, const float, const float, const int);

  private:
    int mLayerIndex;
    std::vector<CACluster> mClusters;
};

inline int CALayer::getLayerIndex() const
{
  return mLayerIndex;
}

inline const std::vector<CACluster>& CALayer::getClusters() const
{
  return mClusters;
}

inline const CACluster& CALayer::getCluster(int clusterIndex) const
{
  return mClusters[clusterIndex];
}

inline int CALayer::getClustersSize() const
{
  return mClusters.size();
}

#endif /* TRACKINGITSU_INCLUDE_CALAYER_H_ */
