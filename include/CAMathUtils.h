/// \file CAMathUtils.h
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

#ifndef TRACKINGITSU_INCLUDE_CAUTILS_H_
#define TRACKINGITSU_INCLUDE_CAUTILS_H_

#include <array>
#include <cmath>

#include "CAConstants.h"

namespace CAMathUtils {
float calculatePhiCoordinate(const float, const float);
float calculateRCoordinate(const float, const float);
GPU_HOST_DEVICE constexpr float getNormalizedPhiCoordinate(const float);
GPU_HOST_DEVICE constexpr float3 crossProduct(const float3&, const float3&);
}

inline float CAMathUtils::calculatePhiCoordinate(const float xCoordinate, const float yCoordinate)
{
  return std::atan2(-yCoordinate, -xCoordinate) + CAConstants::Math::Pi;
}

inline float CAMathUtils::calculateRCoordinate(const float xCoordinate, const float yCoordinate)
{
  return std::sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
}

GPU_HOST_DEVICE constexpr float CAMathUtils::getNormalizedPhiCoordinate(const float phiCoordinate)
{
  return (phiCoordinate < 0) ? phiCoordinate + CAConstants::Math::TwoPi :
         (phiCoordinate > CAConstants::Math::TwoPi) ? phiCoordinate - CAConstants::Math::TwoPi : phiCoordinate;
}

GPU_HOST_DEVICE constexpr float3 CAMathUtils::crossProduct(const float3& firstVector,
    const float3& secondVector)
{

  return float3 { (firstVector.y * secondVector.z) - (firstVector.z * secondVector.y),
      (firstVector.z * secondVector.x) - (firstVector.x * secondVector.z), (firstVector.x * secondVector.y)
          - (firstVector.y * secondVector.x) };
}

#endif /* TRACKINGITSU_INCLUDE_CAUTILS_H_ */
