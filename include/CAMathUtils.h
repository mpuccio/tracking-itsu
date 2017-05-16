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

#include "CAConstants.h"

namespace CAMathUtils {
float calculatePhiCoordinate(const float, const float);
float calculateRCoordinate(const float, const float);
int roundUp(const int, const int);
constexpr float getNormalizedPhiCoordinate(const float);
constexpr std::array<float, 3> crossProduct(const std::array<float, 3>&, const std::array<float, 3>&);
}

constexpr float CAMathUtils::getNormalizedPhiCoordinate(const float phiCoordinate)
{
  return (phiCoordinate < 0) ? phiCoordinate + CAConstants::Math::TwoPi :
         (phiCoordinate > CAConstants::Math::TwoPi) ? phiCoordinate - CAConstants::Math::TwoPi : phiCoordinate;
}

constexpr std::array<float, 3> CAMathUtils::crossProduct(const std::array<float, 3>& firstVector,
    const std::array<float, 3>& secondVector)
{

  return std::array<float, 3> { { (firstVector[1] * secondVector[2]) - (firstVector[2] * secondVector[1]),
      (firstVector[2] * secondVector[0]) - (firstVector[0] * secondVector[2]), (firstVector[0] * secondVector[1])
          - (firstVector[1] * secondVector[0]) } };
}

#endif /* TRACKINGITSU_INCLUDE_CAUTILS_H_ */
