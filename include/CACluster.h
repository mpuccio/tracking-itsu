/// \file CACluster.h
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

#ifndef TRACKINGITSU_INCLUDE_CACLUSTER_H_
#define TRACKINGITSU_INCLUDE_CACLUSTER_H_

#include "CADefinitions.h"
#include "CAMathUtils.h"
#include "CAIndexTableUtils.h"

struct CACluster
    final
    {
      CACluster(const int, const int, const float, const float, const float, const float, const int);
      CACluster(const int, const float3&, const CACluster&);

      float xCoordinate;
      float yCoordinate;
      float zCoordinate;
      float phiCoordinate;
      float rCoordinate;
      int clusterId;
      float alphaAngle;
      int monteCarloId;
      int indexTableBinIndex;
  };

inline CACluster::CACluster(const int clusterId, const int layerIndex, const float xCoordinate, const float yCoordinate,
    const float zCoordinate, const float alphaAngle, const int monteCarloId)
    : xCoordinate { xCoordinate }, yCoordinate { yCoordinate }, zCoordinate { zCoordinate }, phiCoordinate { 0 }, rCoordinate {
        0 }, clusterId { clusterId }, alphaAngle { alphaAngle }, monteCarloId { monteCarloId }, indexTableBinIndex { 0 }
{
  // Nothing to do
}

inline CACluster::CACluster(const int layerIndex, const float3 &primaryVertex, const CACluster& other)
    : xCoordinate { other.xCoordinate }, yCoordinate { other.yCoordinate }, zCoordinate { other.zCoordinate }, phiCoordinate {
        CAMathUtils::getNormalizedPhiCoordinate(
            CAMathUtils::calculatePhiCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y)) }, rCoordinate {
        CAMathUtils::calculateRCoordinate(xCoordinate - primaryVertex.x, yCoordinate - primaryVertex.y) }, clusterId {
        other.clusterId }, alphaAngle { other.alphaAngle }, monteCarloId { other.monteCarloId }, indexTableBinIndex {
        CAIndexTableUtils::getBinIndex(CAIndexTableUtils::getZBinIndex(layerIndex, zCoordinate),
            CAIndexTableUtils::getPhiBinIndex(phiCoordinate)) }
{
  // Nothing to do
}


#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */
