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

#include <initializer_list>
#include <iostream>

CAEvent::CAEvent(const int eventId)
    : mEventId { eventId }
{
  for (int iLayer = 0; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    mLayers[iLayer] = CALayer(iLayer);
  }
}

void CAEvent::addPrimaryVertex(const float xCoordinate, const float yCoordinate, const float zCoordinate)
{
  mPrimaryVertices.emplace_back(std::array<float, 3>{ { xCoordinate, yCoordinate, zCoordinate } });
}

void CAEvent::printPrimaryVertices() const
{
  const int verticesNum = mPrimaryVertices.size();

  for (int iVertex = 0; iVertex < verticesNum; ++iVertex) {

    const std::array<float, 3>& currentVertex = mPrimaryVertices[iVertex];

    std::cout << "-1\t" << currentVertex[0] << "\t" << currentVertex[1] << "\t" << currentVertex[2] << std::endl;
  }
}

void CAEvent::pushClusterToLayer(const int layerIndex, const int clusterId, const float xCoordinate,
    const float yCoordinate, const float zCoordinate, const float aplhaAngle, const int monteCarlo)
{
  mLayers[layerIndex].addCluster(clusterId, xCoordinate, yCoordinate, zCoordinate, aplhaAngle, monteCarlo);
}

const int CAEvent::getTotalClusters() const
{
  int totalClusters = 0;

  for (int iLayer = 0; iLayer < CAConstants::ITS::LayersNumber; ++iLayer) {

    totalClusters += mLayers[iLayer].getClustersSize();
  }

  return totalClusters;
}
