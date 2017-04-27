/// \file CACluster.h
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

#ifndef TRACKINGITSU_INCLUDE_CACLUSTER_H_
#define TRACKINGITSU_INCLUDE_CACLUSTER_H_

class CACluster final
{
  public:
    CACluster(const int, const float, const float, const float, const float, const int);

    const int getClusterId() const;
    const float getXCoordinate() const;
    const float getYCoordinate() const;
    const float getZCoordinate() const;
    const float getAlphaAngle() const;
    const int getMonteCarloId() const;

  private:
    int mClusterId;
    float mXCoordinate;
    float mYCoordinate;
    float mZCoordinate;
    float mAlphaAngle;
    int mMonteCarloId;
};

inline const int CACluster::getClusterId() const
{
  return mClusterId;
}

inline const float CACluster::getXCoordinate() const
{
  return mXCoordinate;
}

inline const float CACluster::getYCoordinate() const
{
  return mYCoordinate;
}

inline const float CACluster::getZCoordinate() const
{
  return mZCoordinate;
}

inline const float CACluster::getAlphaAngle() const
{
  return mAlphaAngle;
}

inline const int CACluster::getMonteCarloId() const
{
  return mMonteCarloId;
}

#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */
