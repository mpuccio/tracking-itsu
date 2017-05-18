/// \file CAMathUtils.cxx
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

#include "CAMathUtils.h"

#include <cmath>

float CAMathUtils::calculatePhiCoordinate(const float xCoordinate, const float yCoordinate)
{
  return std::atan2(-yCoordinate, -xCoordinate) + CAConstants::Math::Pi;
}

float CAMathUtils::calculateRCoordinate(const float xCoordinate, const float yCoordinate)
{
  return std::sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
}

int CAMathUtils::roundUp(const int numToRound, const int multiple)
{
  if (multiple == 0) {

    return numToRound;
  }

  int remainder { numToRound % multiple };

  if (remainder == 0) {

    return numToRound;
  }

  return numToRound + multiple - remainder;
}

int CAMathUtils::findNearestDivisor(const int numToRound, const int divisor) {

  if(numToRound > divisor) {

    return divisor;
  }

  int result = numToRound;

  while(divisor % result != 0) {

    ++result;
  }

  return result;
}
