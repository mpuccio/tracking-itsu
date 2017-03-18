/// \file CAEvent.h
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

#ifndef TRACKINGITSU_INCLUDE_CAEVENT_H_
#define TRACKINGITSU_INCLUDE_CAEVENT_H_

#include <array>
#include <vector>

#include "CAConstants.h"
#include "CAHit.h"

class CAEvent final
{

  public:
    CAEvent(const int);
    const int getEventId() const;
    void setPrimaryVertex(float, float, float);
    void printPrimaryVertex() const;
    const std::array<float, 3>& getPrimaryVertex() const;
    void pushHitToLayer(const int, const float, const float, const float, const int);
    const std::vector<CAHit>& getLayer(const int) const;

  private:
    int mEventId;
    std::array<float, 3> mPrimaryVertex;
    std::array<std::vector<CAHit>, ITSConstants::ITSLayers> mLayerHits;
};

#endif /* TRACKINGITSU_INCLUDE_CAEVENT_H_ */
