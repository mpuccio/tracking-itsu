/// \file CACluster.cxx
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

#include <CACluster.h>

#include <CAIndexTableUtils.h>
#include <CAMathUtils.h>

CACluster::CACluster(const int clusterId, const float xCoordinate, const float yCoordinate, const float zCoordinate,
    const float alphaAngle, const int monteCarloId)
    : clusterId { clusterId }, xCoordinate { xCoordinate }, yCoordinate { yCoordinate }, zCoordinate { zCoordinate }, alphaAngle {
        alphaAngle }, monteCarloId { monteCarloId }, phiCoordinate { 0 }, rCoordinate { 0 }, indexTableBinIndex { 0 }
{
}

CACluster::CACluster(const int layerIndex, const std::array<float, 3> &primaryVertex, const CACluster& other)
    : clusterId { other.clusterId }, xCoordinate { other.xCoordinate }, yCoordinate { other.yCoordinate }, zCoordinate {
        other.zCoordinate }, alphaAngle { other.alphaAngle }, monteCarloId { other.monteCarloId }, phiCoordinate {
        CAMathUtils::getNormalizedPhiCoordinate(
            CAMathUtils::calculatePhiCoordinate(xCoordinate - primaryVertex[0], yCoordinate - primaryVertex[1])) }, rCoordinate {
        CAMathUtils::calculateRCoordinate(xCoordinate - primaryVertex[0], yCoordinate - primaryVertex[1]) }, indexTableBinIndex {
        CAIndexTableUtils::getBinIndex(CAIndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
            CAIndexTableUtils::getPhiBinIndex(phiCoordinate)) }
{
}
