/// \file CAEvent.cxx
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

#include "CAEvent.h"

#include <iostream>

CAEvent::CAEvent(const int eventId)
    : mEventId { eventId }
{
}

void CAEvent::setPrimaryVertex(float xCoordinate, float yCoordinate, float zCoordinate)
{
  mPrimaryVertex[0] = xCoordinate;
  mPrimaryVertex[1] = yCoordinate;
  mPrimaryVertex[2] = zCoordinate;
}

void CAEvent::pushHitToLayer(const int layerIndex, const float xCoordinate, const float yCoordinate,
    const float zCoordinate, const int monteCarlo)
{
  mLayerHits[layerIndex].emplace_back(xCoordinate, yCoordinate, zCoordinate, monteCarlo);
}

inline const int CAEvent::getEventId() const
{
  return mEventId;
}

void CAEvent::printPrimaryVertex() const
{
  std::cout << "-1\t" << mPrimaryVertex[0] << "\t" << mPrimaryVertex[1] << "\t" << mPrimaryVertex[2] << std::endl;
  return;
}

inline const std::array<float, 3>& CAEvent::getPrimaryVertex() const
{
  return mPrimaryVertex;
}

const std::vector<CAHit>& CAEvent::getLayer(const int layerIndex) const
{
  return mLayerHits[layerIndex];
}
