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
#include <vector>

#include "CAConstants.h"
#include "CALayer.h"

class CAEvent final
{

  public:
    explicit CAEvent(const int);

    int getEventId() const;
    const std::array<float, 3>& getPrimaryVertex(const int) const;
    const CALayer& getLayer(const int) const;
    int getPrimaryVerticesNum() const;

    void addPrimaryVertex(const float, const float, const float);
    void printPrimaryVertices() const;
    void pushClusterToLayer(const int, const int, const float, const float, const float, const float, const int);
    const int getTotalClusters() const;

  private:
    const int mEventId;
    std::vector<std::array<float, 3>> mPrimaryVertices;
    std::array<CALayer, CAConstants::ITS::LayersNumber> mLayers;
};

inline int CAEvent::getEventId() const
{
  return mEventId;
}

inline const std::array<float, 3>& CAEvent::getPrimaryVertex(const int vertexIndex) const
{
  return mPrimaryVertices[vertexIndex];
}

inline const CALayer& CAEvent::getLayer(const int layerIndex) const
{
  return mLayers[layerIndex];
}

inline int CAEvent::getPrimaryVerticesNum() const
{

  return mPrimaryVertices.size();
}

#endif /* TRACKINGITSU_INCLUDE_CAEVENT_H_ */
