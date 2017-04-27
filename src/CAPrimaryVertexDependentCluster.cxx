/// \file CAPrimaryVertexDependentCluster.cxx
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

#include "CAPrimaryVertexDependentCluster.h"

#include "CAIndexTableUtils.h"
#include "CAMathUtils.h"

CAPrimaryVertexDependentCluster::CAPrimaryVertexDependentCluster(const int layerIndex,
    const std::array<float, 3>& primaryVertex, const CACluster& cluster)
    : mNativeCluster { &cluster }, mPhiCoordinate { CAMathUtils::getNormalizedPhiCoordinate(
        CAMathUtils::calculatePhiCoordinate(cluster.getXCoordinate() - primaryVertex[0],
            cluster.getYCoordinate() - primaryVertex[1])) }, mRCoordinate { CAMathUtils::calculateRCoordinate(
        cluster.getXCoordinate() - primaryVertex[0], cluster.getYCoordinate() - primaryVertex[1]) }, mIndexTableBinIndex {
        CAIndexTableUtils::getBinIndex(CAIndexTableUtils::getZBinIndex(layerIndex, cluster.getZCoordinate()),
            CAIndexTableUtils::getPhiBinIndex(mPhiCoordinate)) }
{
  // Nothing to do
}
