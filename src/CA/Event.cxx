/// \file Event.cxx
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

#include "ITSReconstruction/CA/Event.h"

#include <iostream>

namespace o2
{
namespace ITS
{
namespace CA
{

Event::Event(const int eventId)
    : mEventId { eventId }
{
  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    mLayers[iLayer] = Layer(iLayer);
  }
}

void Event::addPrimaryVertex(const float xCoordinate, const float yCoordinate, const float zCoordinate)
{
  mPrimaryVertices.emplace_back(float3 { xCoordinate, yCoordinate, zCoordinate });
}

void Event::printPrimaryVertices() const
{
  const int verticesNum { static_cast<int>(mPrimaryVertices.size()) };

  for (int iVertex { 0 }; iVertex < verticesNum; ++iVertex) {

    const float3& currentVertex { mPrimaryVertices[iVertex] };

    std::cout << "-1\t" << currentVertex.x << "\t" << currentVertex.y << "\t" << currentVertex.z << std::endl;
  }
}

void Event::pushClusterToLayer(const int layerIndex, const int clusterId, const float xCoordinate,
    const float yCoordinate, const float zCoordinate, const float aplhaAngle, const int monteCarlo)
{
  mLayers[layerIndex].addCluster(clusterId, xCoordinate, yCoordinate, zCoordinate, aplhaAngle, monteCarlo);
}

int Event::getTotalClusters() const
{
  int totalClusters { 0 };

  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    totalClusters += mLayers[iLayer].getClustersSize();
  }

  return totalClusters;
}

}
}
}
