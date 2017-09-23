/// \file Layer.cxx
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

#include "ITSReconstruction/CA/Layer.h"

#include "ITSReconstruction/CA/Constants.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Layer::Layer()
    : mLayerIndex { Constants::ITS::UnusedIndex }
{
  //Nothing to do
}

Layer::Layer(const int layerIndex)
    : mLayerIndex { layerIndex }
{
  //Nothing to do
}

void Layer::addCluster(const int clusterId, const float xCoordinate, const float yCoordinate, const float zCoordinate,
    const float alphaAngle, const int monteCarlo)
{
  mClusters.emplace_back(clusterId, mLayerIndex, xCoordinate, yCoordinate, zCoordinate, alphaAngle, monteCarlo);
}

}
}
}
