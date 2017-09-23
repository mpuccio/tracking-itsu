/// \file Cluster.cxx
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

#include "ITSReconstruction/CA/Cluster.h"

#include "ITSReconstruction/CA/IndexTableUtils.h"
#include "ITSReconstruction/CA/MathUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Cluster::Cluster(const int clusterId, const int layerIndex, const float xCoordinate, const float yCoordinate,
    const float zCoordinate, const float alphaAngle, const int monteCarloId)
    : xCoordinate { xCoordinate }, yCoordinate { yCoordinate }, zCoordinate { zCoordinate }, phiCoordinate { 0 }, rCoordinate {
        0 }, clusterId { clusterId }, alphaAngle { alphaAngle }, monteCarloId { monteCarloId }, indexTableBinIndex { 0 }
{
  // Nothing to do
}

Cluster::Cluster(const int layerIndex, const float3 &primaryVertex, const Cluster& other)
    : xCoordinate { other.xCoordinate }, yCoordinate { other.yCoordinate }, zCoordinate { other.zCoordinate }, phiCoordinate {
        MathUtils::getNormalizedPhiCoordinate(
            MathUtils::calculatePhiCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y)) }, rCoordinate {
        MathUtils::calculateRCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y) }, clusterId {
        other.clusterId }, alphaAngle { other.alphaAngle }, monteCarloId { other.monteCarloId }, indexTableBinIndex {
        IndexTableUtils::getBinIndex(IndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
            IndexTableUtils::getPhiBinIndex(phiCoordinate)) }
{
  // Nothing to do
}

}
}
}
