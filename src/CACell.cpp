/// \file CACell.cpp
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

#include "CACell.h"

CACell::CACell(const std::array<int, 3>& trackletCoordinates,
  const std::array<float, 3>& normalVectorCoordinates, const float curvature)
  : mTrackletCoordinates{trackletCoordinates},
    mNormalVectorCoordinates{normalVectorCoordinates},
    mCurvature{curvature},
    mLevel{1},
    mNeighboursCount{0}
{
  // Nothing to do
}

CACell::~CACell()
{
  // Nothing to do
}

inline const int CACell::getXCoordinate() const
{
  return mTrackletCoordinates[0];
}

inline const int CACell::getYCoordinate() const
{
  return mTrackletCoordinates[1];
}

inline const int CACell::getZCoordinate() const
{
  return mTrackletCoordinates[2];
}

inline const int CACell::getLevel() const
{
  return mLevel;
}

inline const float CACell::getCurvature() const
{
  return mCurvature;
}

inline const int CACell::getNumberOfNeighbours() const
{
  return mNeighboursCount;
}

inline const std::array<float, 3>& CACell::getNormalVectorCoordinates() const
{
  return mNormalVectorCoordinates;
}

void CACell::setLevel(const int level)
{
  this->mLevel = level;
}

bool CACell::combineCells(const CACell& otherCell)
{
  if(this->getYCoordinate() == otherCell.getZCoordinate() && this->getXCoordinate() == otherCell.getYCoordinate()) {

    mNeighboursCount++;

    int otherCellLevel = otherCell.getLevel();

    if(otherCellLevel >= getLevel()) {

      setLevel(otherCellLevel + 1);
    }

    return true;

  } else {

    return false;
  }
}
