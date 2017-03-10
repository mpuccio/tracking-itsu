/// \file CARoad.cpp
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

/***************************************************************************
 *  Copyright (C) 2017  Iacopo Colonnelli
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#include "CARoad.h"

CARoad::CARoad()
    : mCellIds { }, mRoadSize { }
{
  resetRoad();
}

CARoad::CARoad(int cellLayer, int cellId)
    : CARoad()
{
  setCell(cellLayer, cellId);
}

inline int& CARoad::operator [](const int& i)
{
  return mCellIds[i];
}

inline void CARoad::resetRoad()
{
  mCellIds.fill(sEmptyLayer);
  mRoadSize = 0;
}

void CARoad::setCell(int cellLayer, int cellId)
{
  if (mCellIds[cellLayer] == sEmptyLayer) {

    ++mRoadSize;
  }

  mCellIds[cellLayer] = cellId;
}
