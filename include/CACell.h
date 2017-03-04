/// \file CACell.h
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

#ifndef TRACKINGITSU_INCLUDE_CACELL_H_
#define TRACKINGITSU_INCLUDE_CACELL_H_

#include <array>

class CACell
{
  public:
    CACell(const std::array<int, 3>&, const std::array<float, 3>&, const float);
    virtual ~CACell();

    CACell(const CACell&) = delete;
    CACell &operator=(const CACell &tr) = delete;

    const int getXCoordinate() const;
    const int getYCoordinate() const;
    const int getZCoordinate() const;
    const int getLevel() const;
    const float getCurvature() const;
    const int getNumberOfNeighbours() const;
    const std::array<float, 3>& getNormalVectorCoordinates() const;

    void setLevel(const int level);

    bool combineCells(const CACell&);

  private:
    const std::array<int, 3> mTrackletCoordinates;
    const std::array<float, 3> mNormalVectorCoordinates;
    const float mCurvature;
    int mLevel;
    int mNeighboursCount;
};

#endif /* TRACKINGITSU_INCLUDE_CACELL_H_ */
