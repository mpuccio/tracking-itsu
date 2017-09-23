/// \file Road.cxx
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

#include "ITSReconstruction/CA/Road.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Road::Road()
    : mCellIds { }, mRoadSize { }, mIsFakeRoad { }
{
  resetRoad();
}

Road::Road(int cellLayer, int cellId)
    : Road()
{
  addCell(cellLayer, cellId);
}

void Road::resetRoad()
{
  mCellIds.fill(Constants::ITS::UnusedIndex);
  mRoadSize = 0;
}

void Road::addCell(int cellLayer, int cellId)
{
  if (mCellIds[cellLayer] == Constants::ITS::UnusedIndex) {

    ++mRoadSize;
  }

  mCellIds[cellLayer] = cellId;
}

}
}
}
