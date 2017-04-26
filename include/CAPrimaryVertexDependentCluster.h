/// \file CAPrimaryVertexDependentCluster.h
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

#ifndef TRACKINGITSU_INCLUDE_CACLUSTERWRAPPER_H_
#define TRACKINGITSU_INCLUDE_CACLUSTERWRAPPER_H_

#include <array>

#include "CACluster.h"

class CAPrimaryVertexDependentCluster final
{
public:
    CAPrimaryVertexDependentCluster(const int, const std::array<float, 3>&, const CACluster&);

    const int getClusterId() const;
    const float getXCoordinate() const;
    const float getYCoordinate() const;
    const float getZCoordinate() const;
    const float getAlphaAngle() const;
    const int getMonteCarloId() const;
    const float getPhiCoordinate() const;
    const float getRCoordinate() const;
    const int getIndexTableBinIndex() const;

private:
    const CACluster *mNativeCluster;
    float mPhiCoordinate;
    float mRCoordinate;
    int mIndexTableBinIndex;
};

inline const int CAPrimaryVertexDependentCluster::getClusterId() const
{
  return mNativeCluster->getClusterId();
}

inline const float CAPrimaryVertexDependentCluster::getXCoordinate() const
{
  return mNativeCluster->getXCoordinate();
}

inline const float CAPrimaryVertexDependentCluster::getYCoordinate() const
{
  return mNativeCluster->getYCoordinate();
}

inline const float CAPrimaryVertexDependentCluster::getZCoordinate() const
{
  return mNativeCluster->getZCoordinate();
}

inline const float CAPrimaryVertexDependentCluster::getAlphaAngle() const
{
  return mNativeCluster->getAlphaAngle();
}

inline const int CAPrimaryVertexDependentCluster::getMonteCarloId() const
{
  return mNativeCluster->getMonteCarloId();
}

inline const float CAPrimaryVertexDependentCluster::getPhiCoordinate() const
{
  return mPhiCoordinate;
}

inline const float CAPrimaryVertexDependentCluster::getRCoordinate() const
{
  return mRCoordinate;
}

inline const int CAPrimaryVertexDependentCluster::getIndexTableBinIndex() const
{
  return mIndexTableBinIndex;
}

#endif /* TRACKINGITSU_INCLUDE_CACLUSTERWRAPPER_H_ */
