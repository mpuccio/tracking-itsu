/// \file CAEventLoader.cpp
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

#include "CAEventLoader.h"

#include <sstream>
#include <string>
#include <fstream>

namespace {

constexpr int PrimaryVertexLayerId = -1;
}

std::vector<CAEvent> CAEventLoader::loadEventData(const std::string& fileName)
{
  std::vector<CAEvent> events;
  std::ifstream inputStream;
  std::string line;
  int layerId;
  float xCoordinate, yCoordinate, zCoordinate;

  inputStream.open(fileName);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> layerId >> xCoordinate >> yCoordinate >> zCoordinate) {

      if (layerId == PrimaryVertexLayerId) {

        events.emplace_back(events.size());
        events.back().setPrimaryVertex(xCoordinate, yCoordinate, zCoordinate);

      } else {

        events.back().pushHitToLayer(layerId, xCoordinate, yCoordinate, zCoordinate);
      }
    }
  }

  return events;
}
