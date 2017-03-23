/// \file CAEventLoader.cxx
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
  std::string line, unusedVariable;
  int layerId, monteCarlo;
  int clusterId;
  float xCoordinate, yCoordinate, zCoordinate, alphaAngle;

  inputStream.open(fileName);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> layerId >> xCoordinate >> yCoordinate >> zCoordinate) {

      if (layerId == PrimaryVertexLayerId) {

        events.emplace_back(events.size());
        events.back().setPrimaryVertex(xCoordinate, yCoordinate, zCoordinate);

        clusterId = 0;

      } else {

        if (inputStringStream >> unusedVariable >> unusedVariable >> unusedVariable >> alphaAngle >> monteCarlo) {

          events.back().pushClusterToLayer(layerId, clusterId, xCoordinate, yCoordinate, zCoordinate, alphaAngle,
              monteCarlo);
          ++clusterId;
        }
      }
    }
  }

  return events;
}
