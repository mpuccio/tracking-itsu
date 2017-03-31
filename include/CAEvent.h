/// \file CAEvent.h
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

#ifndef TRACKINGITSU_INCLUDE_CAEVENT_H_
#define TRACKINGITSU_INCLUDE_CAEVENT_H_

#include <array>

#include "CAConstants.h"
#include "CALayer.h"

class CAEvent final
{

  public:
    explicit CAEvent(const int);

    int getEventId() const;
    const std::array<float, 3>& getPrimaryVertex() const;
    float getPrimaryVertexXCoordinate() const;
    float getPrimaryVertexYCoordinate() const;
    float getPrimaryVertexZCoordinate() const;
    const CALayer& getLayer(const int) const;

    void setPrimaryVertex(float, float, float);
    void printPrimaryVertex() const;
    void pushClusterToLayer(const int, const int, const float, const float, const float, const float, const int);
    const int getTotalClusters() const;
    void sortClusters();

  private:
    const int mEventId;
    std::array<float, 3> mPrimaryVertex;
    std::array<CALayer, CAConstants::ITS::LayersNumber> mLayers;
};

inline int CAEvent::getEventId() const
{
  return mEventId;
}

inline const std::array<float, 3>& CAEvent::getPrimaryVertex() const
{
  return mPrimaryVertex;
}

inline float CAEvent::getPrimaryVertexXCoordinate() const
{
  return mPrimaryVertex[0];
}

inline float CAEvent::getPrimaryVertexYCoordinate() const
{
  return mPrimaryVertex[1];
}

inline float CAEvent::getPrimaryVertexZCoordinate() const
{
  return mPrimaryVertex[2];
}

inline const CALayer& CAEvent::getLayer(const int layerIndex) const
{
  return mLayers[layerIndex];
}

#endif /* TRACKINGITSU_INCLUDE_CAEVENT_H_ */
